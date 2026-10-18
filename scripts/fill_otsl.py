#!/usr/bin/env python3
"""Fill missing `otsl` by converting each row's `html`.

OTSL <-> HTML are interconvertible. Sources that ship HTML but no OTSL
(PubTables-1M, CC-OCR) get their OTSL generated here with a worker pool.

Usage: python3 fill_otsl.py <in.parquet> <out.parquet> [--workers N]
"""
import multiprocessing as mp
import re
import sys
from html.parser import HTMLParser

import pyarrow as pa
import pyarrow.parquet as pq

_WS = re.compile(r"\s+")


class OtslBuilder(HTMLParser):
    """HTML -> PaddleOCR OTSL token string."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.cur = None
        self.buf = None

    def _open_row(self):
        if self.cur is None:
            self.cur = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._open_row()
        elif tag in ("td", "th"):
            self._open_row()
            a = dict(attrs)

            def _int(k):
                try:
                    return max(1, int(a.get(k, 1) or 1))
                except (TypeError, ValueError):
                    return 1

            cs, rs = _int("colspan"), _int("rowspan")
            self._pending = ("<fcel>" + "<lcel>" * (cs - 1)
                             + "<ulcel>" * (rs - 1))
            self.buf = []

    def handle_data(self, data):
        if self.buf is not None:
            self.buf.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            text = ""
            if self.buf is not None:
                text = _WS.sub(" ", "".join(self.buf)).strip()
            # 空单元格在 OTSL 里是 <ecel>(而不是 "<fcel>" + 空文本), 但如果它带
            # colspan/rowspan, 跨度 token 要保留: <td colspan=3></td> -> <ecel><lcel><lcel>
            if text:
                head = self._pending
            else:
                head = self._pending.replace("<fcel>", "<ecel>", 1)
            self.cur.append(head + text)
            self.buf = None
            self._pending = None
        elif tag == "tr" and self.cur is not None:
            self.rows.append(self.cur)
            self.cur = None

    def to_otsl(self):
        if self.cur:
            self.rows.append(self.cur)
            self.cur = None
        if not self.rows:
            return ""
        return "<nl>".join(
            "".join(c.strip() for c in r) + "<ecel>" * max(1, len(r))
            for r in self.rows
        )


def html_to_otsl(html):
    if not html:
        return ""
    p = OtslBuilder()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return ""
    return p.to_otsl()


def work(chunk):
    return [html_to_otsl(h) for h in chunk]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    workers = 32
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    out = pq.ParquetWriter(dst, schema, compression="zstd")

    total = filled = 0
    with mp.Pool(workers) as pool:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            htmls = tbl.column("html").to_pylist()
            otsls = tbl.column("otsl").to_pylist()

            todo = [i for i, o in enumerate(otsls) if not o and htmls[i]]
            if todo:
                chunks = [htmls[i] for i in todo]
                n = (len(chunks) + workers * 4 - 1) // (workers * 4)
                pieces = [chunks[k:k + n] for k in range(0, len(chunks), n)]
                for piece, res in zip(pieces, pool.map(work, pieces)):
                    for i, s in zip(todo[:len(piece)], res):
                        otsls[i] = s
                    todo = todo[len(piece):]
                filled += len(todo)

            i = schema.get_field_index("otsl")
            newcol = pa.array(otsls, type=schema.field("otsl").type)
            tbl = tbl.set_column(i, schema.field("otsl"), newcol)
            out.write_table(tbl)
            total += tbl.num_rows
            print(f"  {total} rows, filled {filled}", flush=True)

    out.close()
    print(f"done: {total} rows, {filled} otsl generated -> {dst}")


if __name__ == "__main__":
    main()
