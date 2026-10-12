# What is Presented and What is Published

Data and code to reproduce every number, figure, table, and appendix in

> Cox, D. J. What is presented and what is published: Comparing topic distributions across
> behavior-analytic conferences and journals. *Behavioral Interventions* (manuscript under
> review).

**Interactive supplement:** https://david-j-cox.github.io/verbal-communities-observatory/

The supplement follows any title through each step of the method, shows every title and
subject-area group (Figures 2 and 3, Table 2), and reports the checks behind the approach. It is
built by `scripts/supplement_page.py` and served from `docs/index.html`.

## What the analysis does

The analysis compares 5,327 presentation titles from three behavior-analytic conferences (ABAI,
APBA, SQAB) with 6,872 article titles from fourteen behavior-analytic journals, 2016 through 2026.

1. Each title is turned into a list of 384 numbers with a sentence-transformer model
   (all-MiniLM-L6-v2), so that titles with similar meanings receive similar numbers.
2. For each title, the 75 most similar titles (cosine similarity) form its neighborhood. The
   neighborhood is *presentation-heavy* when presentations are more common in it than in the
   corpus as a whole (43.7%), judged by the lower bound of a 95% confidence interval on the log
   odds.
3. An article is *closely related* to a title when the two are more similar than 99 of every 100
   pairs of titles in the corpus (similarity of at least .525).
4. A title is *flagged* when its neighborhood is presentation-heavy and it has fewer than 10
   closely related articles. 1,552 of the 12,199 titles (12.7%) are flagged.
5. To describe what the flagged titles are about, titles are sorted into 200 subject-area groups
   (k-means), and each group is labeled by a language model. The groups are only a way of
   reporting the results; no title's flag depends on its group.

## Reproducing the results

Requires Python 3.12 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
./run_all.sh
```

`run_all.sh` runs the scripts below in order (about 5 minutes on a recent laptop, plus a
one-time download of the embedding model, about 90 MB). Each script can also be run on its own with
`PYTHONPATH=src:scripts python scripts/<name>.py`. Results are written to `results/`, figures to
`figures/`, and the supplement to `docs/index.html`.

| Script | Produces | In the paper |
|---|---|---|
| `embed_titles.py` | `results/embedding_check.md` (regenerates the vectors and checks them against `data/embeddings/`) | Method: numeric transformation (Table 1) |
| `screen.py` | `results/neighborhood_screen.*`, `results/neighborhood_stability.csv` | Results; Robustness: resampling |
| `subject_groups.py` | `results/subject_summary.csv`, `results/group_map_coords.csv`, `results/subject_groups.md`, `figures/figure2_titles.png`, `figures/figure3_group_map.png` | Table 2, Figures 2 and 3; Robustness: grouping |
| `label_groups.py` | `results/label_check.md` | Method: describing groups |
| `table3_cutoff_sweep.py` | `results/table3_cutoff_sweep.*` | Table 3 |
| `figure4_cutoff_surface.py` | `results/figure4_cutoff_grid.csv`, `figures/figure4_cutoff_surface.png` | Figure 4 |
| `figure1_distance_demo.py` | `figures/figure1_distance_demo.*` | Figure 1 |
| `appendix_a_similarity.py` | `figures/appendix_a_similarity.*` | Appendix A |
| `appendix_b_closely_related.py` | `results/appendix_b_lines.csv`, `figures/appendix_b_closely_related.*` | Appendix B |
| `framing_check.py` | `results/framing_check.*` | Robustness: wording |
| `group_screen_comparison.py` | `results/group_screen_comparison*` | Supplement: Checks tab |
| `supplement_page.py` | `docs/index.html` | Interactive supplement |

The committed `results/`, `figures/`, and `docs/` are the outputs of a full run of
`run_all.sh`, so they can be inspected without rerunning anything.

### What reproduces exactly

* **Embeddings.** The vectors used in the paper are included (`data/embeddings/`), so every
  number reproduces exactly. `embed_titles.py` regenerates them from the titles with the pinned
  model revision (`1110a243`) and checks them against the included ones; they match to within
  about 2 × 10⁻⁷ (floating-point rounding). The included vectors are used because that rounding
  can tip a few comparisons that sit exactly at a cutoff: rebuilt entirely from regenerated
  vectors, one title of 12,199 changes whether its neighborhood is presentation-heavy (4,940
  against 4,939), two titles gain or lose one closely related article, and no flag changes.
* **Flags, counts, tables, and figures.** Finding a title's nearest neighbors involves no random
  starting point, and every random step (k-means, resampling, multidimensional scaling) uses a
  fixed seed.
* **Group labels.** A language model's answers cannot be regenerated exactly, so its inputs and
  outputs are archived in `data/labels/` (see below). `label_groups.py` rebuilds the evidence
  and prompts from the data, checks that they match the archive, and replays the archived
  responses; it calls the model only when run with `--call-model`, and only for responses that
  are missing from the archive.

## Data

All files in `data/` are committed (about 77 MB, most of it the embeddings).

### `data/titles/` (one Parquet file per source)

| File | Titles | Contents |
|---|---:|---|
| `conference.parquet` | 4,443 | ABAI annual convention presentations, 2016-2026 |
| `apba.parquet` | 252 | APBA annual convention presentations, 2023-2026 |
| `sqab.parquet` | 632 | SQAB annual meeting presentations (129 talks, 503 posters), 2016-2026 |
| `journal.parquet` | 15,536 | Articles in eight general behavior-analysis journals, all years (4,408 from 2016-2026) |
| `specialty.parquet` | 7,936 | Articles in six specialty journals, all years (2,464 from 2016-2026) |
| `conference_fit.parquet` | 4,458 | ABAI titles as first extracted, used only to fit the subject-area groups |
| `journal_fit.parquet` | 15,594 | General-journal titles as first retrieved, used only to fit the subject-area groups |

The general journals are *Behavior Analysis in Practice*, *Behavior Analysis: Research and
Practice*, *Behavior Modification*, *Behavioral Interventions*, the *Journal of Applied Behavior
Analysis*, the *Journal of the Experimental Analysis of Behavior*, *The Analysis of Verbal
Behavior*, and *The Behavior Analyst* / *Perspectives on Behavior Science*. The specialty
journals are *Behavior and Social Issues*, *Education and Treatment of Children*, the *Journal of
Behavioral Education*, the *Journal of Contextual Behavioral Science*, the *Journal of
Organizational Behavior Management*, and *The Psychological Record*.

Conference titles were extracted from the published convention and meeting programs; article
titles were retrieved from OpenAlex. Non-research items (e.g., awards, business meetings, social
events, journal front matter, supplementary-material deposits, and memorials) were removed with
an anchored text filter before analysis, so the tables contain only the analyzed titles. The
columns are a record identifier, the year, the title, the journal (articles), and, for ABAI, the
program's own event fields.

The two `_fit` tables hold the titles the 200 groups were fit on, as they stood at that time.
They differ slightly from the analyzed tables (e.g., 76 ABAI titles that the program extraction
had run together with a chair's name were later repaired). Fitting on these exact titles
reproduces the groups reported in the paper.

### `data/embeddings/` (the vectors used in the paper)

One `<table>__all-minilm-l6-v2.npy` file per title table: a float32 array with one row of 384
numbers per title, in the same order as `data/titles/<table>.parquet`.

### `data/labels/` (archived group labels)

* `group_labels.csv`: for each of the 199 groups, the label, definition, and how it differs from
  its neighbors, with the model, prompt version, and SHA-256 hashes of the prompt and evidence.
* `evidence.jsonl`: the evidence given to the model for each group (key terms, the 10
  presentation and 10 article titles closest to the group center, and the three most similar
  groups).
* `prompts/<group>.txt`: the exact prompt sent for each group.
* `responses/<sha256>.json`: every raw model response, keyed by the SHA-256 of the exchange.

Labels were generated by Claude Opus 5.5 (Anthropic) and reviewed by the author before use.

## Repository layout

```
data/titles/        analyzed titles (inputs)
data/embeddings/    the vectors used in the paper
data/labels/        archived group labels, evidence, prompts, and responses
src/presented_published/
    neighborhood.py the two conditions and the flag
    groups.py       subject-area groups, synonym merge, key terms
    labeling.py     the labeling prompt and response validation
    style.py        figure style
scripts/            one script per result (see table above)
tests/              unit tests (python -m pytest)
results/ figures/   outputs of run_all.sh
docs/index.html     the interactive supplement (GitHub Pages)
```

## License and citation

Code is released under the MIT License (`LICENSE`). Data are released under the Creative Commons
Attribution 4.0 International license (`data/LICENSE.md`). If you use this repository, please
cite the article above; `CITATION.cff` has the details.

Contact: David J. Cox, dcox@endicott.edu
