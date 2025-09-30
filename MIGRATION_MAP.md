# MIGRATION_MAP

업로드된 노트북에서 확인된 주요 함수/로직을 첨부 git 레이아웃에 맞춰 분리했습니다.

| Notebook Function / Concept       | New Module / Function                                      |
|-----------------------------------|-------------------------------------------------------------|
| `load_csv`, `standardize_columns` | `src/common/io.py` → `load_csv`, `standardize_columns`     |
| `pick_col`                        | `src/common/io.py` → `pick_col`                            |
| `detect_timestamp_column`         | `src/common/io.py` → `detect_timestamp_column`             |
| `parse_time_index`, `ensure_hourly_index` | `src/common/io.py` / `src/data_process/preprocess.py` |
| `aggregate_to_hourly`, `aggregate_on_duplicate_timestamps` | `src/data_process/preprocess.py` |
| `coerce_numeric`                  | `src/common/io.py`                                         |
| `default_day_mask`                | `src/common/metrics.py`                                    |
| `compute_denom`                   | `src/common/metrics.py`                                    |
| `nmae_with_denom`                 | `src/common/metrics.py`                                    |
| `nmae(KPX)`                       | `src/common/metrics.py` → `nmae_kpx`                       |
| `make_features`                   | `src/data_process/features.py`                             |
| `fit_residual_model`              | `src/model/train_xgb.py` (XGB 학습/추론으로 단순화)        |
| `tune_w_on_train`                 | (옵션) `src/model/ensembles.py` 내부 로직으로 확장 가능     |
| `plot_range_df`, `_do_plot`       | `src/visualize/plots.py`                                   |

> 주: 노트북의 그래프/EDA 셀은 `notebooks/` 폴더로 이동 권장.
