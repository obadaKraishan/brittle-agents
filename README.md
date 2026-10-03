# Brittle Agents

Code and data for **Loud Failures, Quiet Failures: Fault Detection and Recovery
in Tool-Using Language Model Agents** (Obada Kraishan, IEEE CogMI 2026, to
appear).

The study wraps the executable multi-turn environments of the Berkeley
Function-Calling Leaderboard (BFCL) in a fault-injection layer. One typed fault
(timeout, missing tool, schema drift, or silent corruption) fires once per
trial, and each trial is scored for whether the agent noticed the failure,
changed its next action, reached the state of an independent fault-free run,
and called the same tool repeatedly. Six models from three families (matched
reasoning and instruct pairs) ran 1,920 trials over 24 tasks.

## Reproducing the paper

Requires Python 3.10 to 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` rebuilds every number, table, and figure from the released
trajectories and the cached judge codings. It makes no API calls and takes
well under a minute. The last step, `tests/verify_paper_numbers.py`, rebuilds
all 1,920 trials independently of the pipeline, checks them against the scored
data, and recomputes each number reported in the paper.

Options:

- `python reproduce.py --self-test` first fetches the BFCL environment code at
  the pinned commit (from GitHub, into `third_party/`) and runs the
  fault-injection self-test on all 24 tasks.
- `python reproduce.py --tex path/to/main.tex` also compares every row of the
  paper's data tables with the recomputed values and checks that the generated
  table files are identical to the tables in the paper source.

## Where each table and figure comes from

Table numbers follow the order of the tables in the paper. Table I (related
work) contains no data.

| Paper | Content | Script | Output |
|---|---|---|---|
| Table II | Conditions and how often each fault fired | `06_figures_tables/make_tables.py` | `06_figures_tables/outputs/tables/tab_faults.tex` |
| Table III | Model panel | `06_figures_tables/make_tables.py` | `06_figures_tables/outputs/tables/tab_design.tex` |
| Table IV | Detection rates (RQ1) | `05_analysis/rq1_detection/rq1_analysis.py` → `make_tables.py` | `06_figures_tables/outputs/tables/tab_rq1.tex` |
| Table V | Reasoning vs instruct siblings (RQ2) | `05_analysis/rq2_reasoning_recovery/rq2_analysis.py` → `make_tables.py` | `06_figures_tables/outputs/tables/tab_rq2.tex` |
| Table VI | Recovery and effort by condition (RQ3) | `05_analysis/rq3_fault_costs/rq3_analysis.py` → `make_tables.py` | `06_figures_tables/outputs/tables/tab_rq3.tex` |
| Fig. 1 | Detection by failure visibility | `06_figures_tables/make_figures.py` | `06_figures_tables/outputs/figures/fig1_detection.pdf` |
| Fig. 2 | Reasoning minus instruct differences | `06_figures_tables/make_figures.py` (reads RQ2 results) | `06_figures_tables/outputs/figures/fig2_reasoning.pdf` |
| Fig. 3 | Recovery against the fault-free baseline; repeated calls | `06_figures_tables/make_figures.py` | `06_figures_tables/outputs/figures/fig3_consequences.pdf` |

Every table also has a Markdown copy (`.md`) next to the `.tex` file. The
figure script also draws `fig0_method`, a schematic not used in the paper.

Numbers reported in the text:

| Paper section | Output |
|---|---|
| Method: task filtering (400 candidates, 312 eligible, 24 sampled) | `01_environment/outputs/suite_summary.txt` |
| Method: self-test | `01_environment/outputs/fault_selftest_report.txt` |
| Method: agreement of judge and hand-coding | `04_codebook/outputs/judge_agreement.txt` |
| Method: data quality (false-alarm baseline, step cap, tool calls written as text) | `03_data_processing/outputs/scoring_report.txt`, `validation_report.txt` |
| Method: when the fault fired | `05_analysis/rq3_fault_costs/outputs/rq3_report.txt` |
| RQ1 | `05_analysis/rq1_detection/outputs/rq1_report.txt` |
| RQ2 | `05_analysis/rq2_reasoning_recovery/outputs/rq2_report.txt` |
| RQ3 | `05_analysis/rq3_fault_costs/outputs/rq3_report.txt` |
| Secondary analysis (instruction to check each result) | `05_analysis/secondary_scaffolding/outputs/scaffolding_report.txt` |

Each analysis folder also writes its results as CSV and JSON.

## Repository layout

```
config.yaml                 design, models, paths, pinned BFCL versions
reproduce.py                runs the whole analysis pipeline offline
00_setup/                   environment check
01_environment/             task-suite builder, fault-injection layer, self-test
  outputs/                  frozen 24-task suite and build log
02_data_collection/         agent runner (needs an API key)
  outputs/trajectories/     raw trajectories, one JSON line per trial
03_data_processing/         scoring and validation
  outputs/                  master_trials.csv (one row per trial), judge cache
04_codebook/                variable codebook and judge-human agreement
  outputs/                  codebook, hand-coded sample
05_analysis/                RQ1, RQ2, RQ3, and the secondary analysis
06_figures_tables/          the paper's tables and figures
tests/                      independent check of the paper's numbers
```

## Data

- `02_data_collection/outputs/trajectories/`: the full transcript of every
  trial (messages, tool calls, tool results, fault markers, end state),
  one file per model, condition, and prompt arm; 16 MB in total.
  `reference_states.json` holds the end state of each task's ground-truth
  solution.
- `03_data_processing/outputs/master_trials.csv`: one row per trial with all
  measures; `04_codebook/outputs/codebook.md` documents every column.
- `03_data_processing/outputs/judge_cache.jsonl`: the judge model's coding of
  each trial, used by the scoring step instead of new API calls.
- `04_codebook/outputs/handcode_sample_FILLED.csv`: 100 trials coded by hand,
  used for the agreement figures.

A seventh and eighth model, a free-tier Nemotron reasoning and instruct pair,
were dropped before analysis after repeated rate-limit failures; they have no
completed trials and are not included.

## Collecting new data

New runs need an OpenRouter API key in `.env` (copy `.env.example`).

```bash
python 00_setup/setup_check.py --ping                  # environment and key
python 01_environment/build_task_suite.py --overwrite  # rebuild the suite
python 01_environment/fault_injection.py --self-test
python 02_data_collection/run_agents.py --smoke-test   # 2 tasks, 1 seed
python 02_data_collection/run_agents.py                # full grid
python 03_data_processing/score_trajectories.py        # codes new trials with the judge
```

The task suite is built from BFCL dataset revision
`61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` and environment code at gorilla
commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. These versions reproduce
`task_suite.json` and `reference_states.json` exactly. Model sampling is not
seeded, so new runs give new trajectories.

## License

The code is released under the MIT License (`LICENSE`). Files derived from the
Berkeley Function-Calling Leaderboard (the task suite, reference states,
trajectories, and hand-coded sample) are distributed under the Apache License
2.0 (`LICENSE-DATA`); `NOTICE` lists these files, gives attribution, and
describes the changes made.

## Citation

See `CITATION.cff`, or cite:

> O. Kraishan, "Loud Failures, Quiet Failures: Fault Detection and Recovery in
> Tool-Using Language Model Agents," in *Proc. IEEE International Conference
> on Cognitive Machine Intelligence (CogMI)*, 2026, to appear.
