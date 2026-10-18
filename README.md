# PaddleOCR-VL Table Recognition

Benchmark construction, scoring, and fine-tuning pipeline for table structure
recognition with PaddleOCR-VL.

- **Merged benchmark** — [`CheneyWong/table_recognition_eval`](https://huggingface.co/datasets/CheneyWong/table_recognition_eval)
  on Hugging Face: 184,074 tables from 17 sources, packed into 64 shards
  balanced by image bytes.
- **Source corpora** — individual datasets published under
  [`CheneyWong`](https://huggingface.co/datasets/CheneyWong/datasets).

## Benchmark

`table_recognition_eval` merges the source corpora below into one evaluation
set. Every row carries both an **OTSL** and an **HTML** ground truth, plus a
`structure_only` flag marking tables that have no cell text at all.

| benchmark | rows | origin |
|---|---:|---|
| `pubtables1m` | 93,834 | PubTables-1M, test split |
| `mmsci` | 12,129 | MMSci-Table, test split |
| `synthtabnet` | 29,990 | SynthTabNet, test |
| `fintabnet` | 10,397 | FinTabNet_OTSL, test |
| `pubtabnet_icdar21` | 9,064 | ICDAR 2021 official challenge test set |
| `pubtablesv2` | 7,680 | PubTables-v2, test |
| `tableverse` | 4,951 | TableVerse-5K |
| `tabrecset_en` | 4,394 | TabRecSet (English) |
| `hitab` | 3,597 | HiTab |
| `scitsr` | 3,000 | SciTSR |
| `sprint_mustard` | 1,429 | SPRINT-MUSTARD |
| `tabrecset_zh` | 1,065 | TabRecSet (Chinese) |
| `tablet` | 909 | TABLET, test split |
| `omnidocbench16` | 665 | OmniDocBench v1.6 |
| `omnidocbench15` | 512 | OmniDocBench v1.5 |
| `ccocr` | 300 | CC-OCR |
| `ocrbenchv2` | 158 | OCRBench-v2 table-parsing tasks with explicit bbox |
| **total** | **184,074** | |

Shards are balanced by **total image bytes**, not row count: each holds
roughly 135 MB of images and mixes 15–17 different sources, so a single
shard already gives broad coverage.

Over-long tables are kept out of the 64 so they do not skew that balance;
they ship separately as `overlong.parquet`.

## Source corpora

Published separately so you can pull only what you need:

| dataset | contents |
|---|---|
| [`FinTabNet-OTSL`](https://huggingface.co/datasets/CheneyWong/FinTabNet-OTSL) | financial statement tables with OTSL annotations |
| [`HiTab`](https://huggingface.co/datasets/CheneyWong/HiTab) | hierarchical tables with dense row/column spans |
| [`MMSci-Table`](https://huggingface.co/datasets/CheneyWong/MMSci-Table) | scientific tables from recent papers |
| [`PubTabNet`](https://huggingface.co/datasets/CheneyWong/PubTabNet) | the original PubTabNet release |
| [`PubTabNet-ICDAR-test-GT`](https://huggingface.co/datasets/CheneyWong/PubTabNet-ICDAR-test-GT) | official ICDAR 2021 challenge ground truth |
| [`PubTables-1M`](https://huggingface.co/datasets/CheneyWong/PubTables-1M) | 1M scientific tables with structure annotations |
| [`PubTables-v2`](https://huggingface.co/datasets/CheneyWong/PubTables-v2) | PubTabNet and PubTables-1M combined |
| [`SPRINT-MUSTARD`](https://huggingface.co/datasets/CheneyWong/SPRINT-MUSTARD) | structure-only tables, 13 languages |
| [`SynthTabNet`](https://huggingface.co/datasets/CheneyWong/SynthTabNet) | synthetic tables |
| [`TABLET`](https://huggingface.co/datasets/CheneyWong/TABLET) | table QA benchmark, 9 subsets |
| [`TableVerse-5K`](https://huggingface.co/datasets/CheneyWong/TableVerse-5K) | 5K complex Chinese tables |
| [`SciTSR`](https://huggingface.co/datasets/CheneyWong/SciTSR) | SciTSR train/val/test tarballs |
| [`rebuilt_table_sources`](https://huggingface.co/datasets/CheneyWong/rebuilt_table_sources) | the four rebuilt sources on their own: OmniDocBench v1.5/v1.6, CC-OCR, OCRBench-v2 tables |

### Rebuilt sources

Four of the sources above (`omnidocbench15`, `omnidocbench16`, `ccocr`,
`ocrbenchv2`) are not published as image+html pairs and had to be rebuilt from
upstream annotations:

- **OmniDocBench v1.5 / v1.6** — the release ships page images plus a JSON whose
  table entries carry a `poly` and an `html` field. Each `category_type ==
  "table"` entry is cropped with its `poly` plus a 10px margin, and that entry's
  `html` becomes the ground truth. The public `opendatalab/OmniDocBench`
  repository publishes only the v1.5 JSON.
- **CC-OCR** — HTML ships with the release; OTSL is derived from it.
- **OCRBench-v2** — a general document-VQA benchmark, mostly unusable for table
  recognition. Of its 700 `table parsing` tasks only the 158 that name an
  explicit region in the prompt have a croppable table. Ground truth is
  `answers[0]`; 19 of those rows are Markdown rather than HTML.

Those 1,635 tables are published separately as
[`rebuilt_table_sources`](https://huggingface.co/datasets/CheneyWong/rebuilt_table_sources).

## Scoring

`TEDS = 1 - TD / max(N_pred, N_gt)` over the table tree, where `TD` is the
tree edit distance and `N` the node count. `TEDS-S` is the structure-only
variant: cell text is erased, spans are kept.

The two are used differently:

- **`structure_only == true`** — every cell is empty and only the grid is
  annotated (31,429 rows, 17%). The text half of TEDS degenerates into
  "did the model correctly output nothing", so **TEDS is not computed at
  all** for these rows; only TEDS-S is.
- **otherwise** — both are computed. TEDS hitting 1.0 implies TEDS-S is also
  1.0, so the structure metric is short-circuited in that case.

Either way at most one tree-edit-distance pass runs per row.

Measured round-trip fidelity between the two dialects, on a 2,500-row sample:

| check | result |
|---|---|
| byte-identical `otsl -> html -> otsl` | 41.2% |
| identical after normalisations below | 72.8% |
| span tokens (`colspan`/`rowspan`) fully preserved | 91.0% |

Conversion is structurally faithful but not byte-stable. Short rows are padded
when written as HTML (OTSL allows ragged rows), an empty cell can be spelled
`<fcel>` or `<ecel>`, and vertical merges on irregular layouts are re-derived.

## vLLM environment

Evaluating PaddleOCR-VL-1.6 with vLLM needs the offline image
`paddleocr-genai-vllm-server:latest-nvidia-gpu-offline`. The tarball is
**7.34 GB**, too large for GitHub, so it is hosted on Baidu Netdisk:

```
file    : paddleocrvl-docker
link    : https://pan.baidu.com/s/1Vwt9XtYP5aBPjfdXzxBRAA?pwd=y8na
code    : y8na
sha256  : f40e1d785c7c09c29ff1118cdf7edffec500459c3964ca05c419657d2107a364
```

```bash
sha256sum paddleocr-vllm-offline.tar    # verify against the digest above

docker load -i paddleocr-vllm-offline.tar
docker run --rm --gpus all --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8080 --backend vllm
```

## Scripts

`scripts/` holds the two ground-truth converters referenced by the dataset
card. Both walk a parquet file row group by row group and convert across a
worker pool, so memory stays bounded and `--workers 128` is safe on a
many-core machine.

```bash
python scripts/fill_otsl.py shard_0000.parquet --workers 128   # html -> otsl
python scripts/fill_html.py shard_0000.parquet --workers 128   # otsl -> html
```

## Citation

Please cite the original datasets you use. These repositories only re-package
them under a unified schema.
