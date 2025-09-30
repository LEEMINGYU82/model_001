# gen_rate_ported

이 리포는 업로드하신 노트북(`gen_rate_train_save_test_v2_4.ipynb`)을
첨부 Git(`pv_pred_da_001-master`)의 **구조/흐름**과 유사한 형태로 **모듈화/CLI화**한 포팅 템플릿입니다.

## 구조

```
gen_rate_ported/
├─ main.py                  # CLI 진입점 (train / predict / eval)
├─ configs/
│  └─ config.yaml           # 데이터 컬럼/경로/하이퍼파라미터 설정
├─ src/
│  ├─ common/
│  │  ├─ constants.py       # 경로/상수
│  │  ├─ logger.py          # rich 로거
│  │  ├─ io.py              # CSV 로드/컬럼 표준화/시간 파싱 등 (노트북의 I/O 유틸)
│  │  └─ metrics.py         # nMAE 등 지표 (denominator 전략 포함)
│  ├─ data_process/
│  │  ├─ preprocess.py      # 전처리/리샘플/중복타임스탬프 집계
│  │  └─ features.py        # 파생변수 생성(make_features)
│  ├─ model/
│  │  ├─ ensembles.py       # MeanEnsemble (필요시)
│  │  └─ train_xgb.py       # XGBoost 학습/저장/로드/추론
│  └─ visualize/
│     └─ plots.py           # 실제 vs 예측 라인 플롯(utils)
├─ models/                  # 학습된 모델/스케일러 저장
├─ outputs/                 # 예측/평가 결과
├─ logs/                    # 로그
├─ scripts/
│  ├─ run_train.sh          # 학습 실행 예시
│  └─ run_predict.sh        # 예측/평가 실행 예시
└─ requirements.txt
```

## 빠른 시작

```bash
# 1) 패키지 설치
pip install -r requirements.txt

# 2) 설정 확인/수정
vim configs/config.yaml

# 3) 학습
python main.py train --train_csv /path/to/train.csv --out_dir ./models

# 4) 예측
python main.py predict --test_csv /path/to/test.csv --model_dir ./models --out_csv ./outputs/preds.csv

# 5) 평가 (nMAE, nMAE(KPX))
python main.py eval --truth_csv /path/to/test.csv --pred_csv ./outputs/preds.csv --report ./outputs/metrics.json
```

## 노트북 → 모듈 매핑 (요약)

- I/O/컬럼 탐지/시간 파싱: `src/common/io.py`
- 시간 집계/보간/중복 처리: `src/data_process/preprocess.py`
- 특성 생성(`make_features`): `src/data_process/features.py`
- 지표(`compute_denom`, `nmae_with_denom`, `default_day_mask`, `nmae_kpx`): `src/common/metrics.py`
- 시각화(`plot_range_df`): `src/visualize/plots.py`
- 모델 학습/추론(`fit_residual_model` 유사 XGBoost 학습): `src/model/train_xgb.py`
- 평균 앙상블(`MeanEnsemble`): `src/model/ensembles.py`

세부 매핑은 `MIGRATION_MAP.md` 참고.
