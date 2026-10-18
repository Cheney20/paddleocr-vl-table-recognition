#!/usr/bin/env python3
"""Fill missing `html` by converting each row's `otsl` (the reverse direction).

OTSL and HTML are interconvertible. Sources that ship OTSL but no HTML
(TabRecSet) get HTML generated here.

Token grammar handled:
    <fcel>            new cell starts, text follows inline
    <ecel>            empty cell
    <lcel>            cell continues to the LEFT  -> colspan += 1
    <ulcel>           cell continues ABOVE      -> rowspan += 1
    <xcel>            continues both ways       -> colspan += 1, rowspan += 1
    <nl>              end of row

Rows shorter than the widest row are padded with empty cells, matching the
square-grid convention PaddleOCR uses.

Usage: python3 fill_html.py <in.parquet> <out.parquet> [--workers N]
"""
import multiprocessing as mp
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_TOK = re.compile(r"<(fcel|ecel|lcel|ulcel|xcel|nl)>")
_WS = re.compile(r"\s+")

_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def _esc(t):
    return "".join(_ESC.get(c, c) for c in t)


def parse_otsl(otsl):
    """-> list of rows; each row is a list of (text, colspan, rowspan)."""
    rows, cur = [], []
    cell = None          # [text, colspan, rowspan] of the cell being filled

    def flush():
        nonlocal cell
        if cell is not None:
            cur.append((cell[0].strip(), cell[1], cell[2]))
            cell = None

    pos = 0
    for m in _TOK.finditer(otsl or ""):
        text = otsl[pos:m.start()]
        pos = m.end()
        tag = m.group(1)
        if cell is not None:
            cell[0] += text
        if tag == "fcel":
            flush()
            cell = ["", 1, 1]
        elif tag == "ecel":
            flush()
            cur.append(("", 1, 1))
        elif tag == "lcel":
            # continuation extends the *current* cell if one is open,
            # otherwise the last one already committed to this row
            if cell is not None:
                cell[1] += 1
            elif cur:
                t, cs, rs = cur[-1]
                cur[-1] = (t, cs + 1, rs)
        elif tag == "ulcel":
            if cell is not None:
                cell[2] += 1
            elif cur:
                t, cs, rs = cur[-1]
                cur[-1] = (t, cs, rs + 1)
        elif tag == "xcel":
            if cell is not None:
                cell[1] += 1
                cell[2] += 1
            elif cur:
                t, cs, rs = cur[-1]
                cur[-1] = (t, cs + 1, rs + 1)
        elif tag == "nl":
            flush()
            rows.append(cur)
            cur = []
    flush()
    if cur:
        rows.append(cur)
    return rows


def otsl_to_html(otsl):
    """Rebuild a <table> from an OTSL token string."""
    rows = parse_otsl(otsl)
    rows = [r for r in rows if r]
    if not rows:
        return ""
    width = max(sum(c[1] for c in r) for r in rows) or 1

    body = []
    for r in rows:
        tds, used = [], 0
        for text, cs, rs in r:
            used += cs
            attrs = ""
            if cs > 1:
                attrs += f' colspan="{cs}"'
            if rs > 1:
                attrs += f' rowspan="{rs}"'
            tds.append(f"<td{attrs}>{_esc(text)}</td>")
        while used < width:                     # pad short rows
            tds.append("<td></td>")
            used += 1
        body.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>" + "".join(body) + "</table>"


def work(chunk):
    return [otsl_to_html(o) for o in chunk]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 128

    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    writer = pq.ParquetWriter(dst, schema, compression="zstd")
    total = filled = 0
    with mp.Pool(workers) as pool:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            htmls = t.column("html").to_pylist()
            otsls = t.column("otsl").to_pylist()
            todo = [i for i in range(len(otsls)) if not htmls[i] and otsls[i]]
            if todo:
                for i, s in zip(todo, pool.map(work, [otsls[i] for i in todo],
                                                chunksize=64)):
                    htmls[i] = s
                filled += len(todo)
            i = schema.get_field_index("html")
            t = t.set_column(i, schema.field("html"),
                             pa.array(htmls, type=schema.field("html").type))
            writer.write_table(t)
            total += t.num_rows
    writer.close()
    print(f"done: {total} rows, {filled} html generated -> {dst}")


if __name__ == "__main__":
    main()
