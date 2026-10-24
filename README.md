# Amazon M2 Semantic Session-Based Recommender
# Amazon M2 의미 기반 세션 추천 시스템

A multilingual session-based recommendation system built on the Amazon M2 dataset and evaluated against the full catalog of 1,393,956 products.

Amazon M2 데이터셋을 기반으로 구축한 다국어 세션 기반 추천 시스템이며, 최종 평가는 1,393,956개의 전체 상품 카탈로그를 대상으로 수행했습니다.

## Project Summary | 프로젝트 요약

The system predicts the next product in a shopping session using frozen multilingual semantic product embeddings and a Transformer-based sequential recommender.

이 시스템은 동결된 다국어 상품 semantic embedding과 Transformer 기반 sequential recommender를 사용하여 세션의 다음 상품을 예측합니다.

**Final model | 최종 모델**: 4-Layer Semantic SASRec  
**Full-catalog Recall@20 | 전체 카탈로그 Recall@20**: 0.425167  
**Full-catalog NDCG@20 | 전체 카탈로그 NDCG@20**: 0.193198  
**Full-catalog MRR@20 | 전체 카탈로그 MRR@20**: 0.125286  
**Catalog size | 카탈로그 크기**: 1,393,956 products  
**Test sessions | 테스트 세션 수**: 360,625

---

## 1. Problem Definition | 문제 정의

Given a session of previously interacted products:

이전 상호작용 상품으로 구성된 세션이 주어졌을 때:

```text
[item_1, item_2, ..., item_n]
```

the model predicts the next product:

모델은 다음 상품을 예측합니다:

```text
item_(n+1)
```

The final evaluation ranks the true target against the complete catalog, not only sampled negatives.

최종 평가는 sampled negative가 아니라 전체 상품 카탈로그를 대상으로 실제 target의 순위를 평가합니다.

---

## 2. Dataset | 데이터셋

Amazon M2 contains multilingual product metadata and session-based behavioral data across six locales:

Amazon M2는 6개 locale의 다국어 상품 메타데이터와 세션 행동 데이터를 포함합니다.

```text
DE, JP, UK, ES, FR, IT
```

Key statistics | 주요 통계:

| Statistic | Value |
|---|---:|
| Product metadata rows | 1,551,057 |
| Unique ASINs | 1,410,675 |
| Sessions | 3,606,249 |
| Behavioral products used by the model | 1,393,956 |
| Train sessions | 2,884,999 |
| Validation sessions | 360,625 |
| Test sessions | 360,625 |

Important EDA findings included locale imbalance, duplicated sessions, multilingual metadata, strong price skew, and a systematic abnormal price value of `40,000,000.07`.

EDA에서는 locale 불균형, 중복 세션, 다국어 메타데이터, 강한 가격 왜도, 그리고 `40,000,000.07`의 반복적인 비정상 가격 값을 확인했습니다.

---

## 3. Pipeline | 파이프라인

```text
Raw Product + Session Data
        ↓
String / Missing-Value Normalization
        ↓
Session Parsing
        ↓
Behavioral Vocabulary Construction
        ↓
ASIN → Numerical Item ID
        ↓
Train / Validation / Test Split
        ↓
Locale OHE + Locale-wise Price Processing
        ↓
Product Text Construction
        ↓
Voyage-4-Large Semantic Embeddings
        ↓
Multi-Locale Embedding Aggregation
        ↓
Session Dataset + Dynamic Padding
        ↓
Sequential Recommendation Model
        ↓
Full-Catalog Evaluation
```

The item ID `0` is reserved for padding.

상품 ID `0`은 padding 전용으로 예약됩니다.

---

## 4. Semantic Product Representation | 상품 의미 표현

Product text is constructed from available metadata fields such as title, brand, color, size, model, material, author and description. Missing fields are omitted instead of being represented as literal `NaN` tokens.

상품 텍스트는 title, brand, color, size, model, material, author, description 중 실제로 존재하는 필드만 사용하여 구성합니다. 누락 필드는 `NaN` 문자열로 넣지 않고 제외합니다.

Semantic encoder | 의미 임베딩 모델:

```text
Voyage-4-Large
Embedding dimension: 1024
```

Multiple locale-specific representations of the same ASIN are embedded separately and aggregated into a single global product representation.

동일 ASIN의 여러 locale 표현은 각각 임베딩한 후 하나의 global product representation으로 통합합니다.

Final frozen embedding matrix | 최종 동결 임베딩 행렬:

```text
1,393,957 × 1024
```

Row 0 is the PAD vector.

0번 행은 PAD 벡터입니다.

---

## 5. Final Architecture | 최종 아키텍처

```text
Item IDs
   ↓
Frozen 1024-D Semantic Embeddings
   ↓
Linear Projection + GELU + LayerNorm
1024 → 128
   ↓
+ Positional Embeddings
+ Locale Conditioning
   ↓
4-Layer Transformer Encoder
   ↓
Final Session Representation (128-D)
   ↓
Dot Product with Product Vectors
   ↓
Top-K Recommendation
```

The Transformer uses causal attention and padding masks.

Transformer는 causal attention과 padding mask를 사용합니다.

---

## 6. Training | 학습

Training uses sampled negatives for computational efficiency.

계산 효율성을 위해 학습 시 sampled negative를 사용합니다.

For each positive target:

각 positive target마다:

```text
1 positive item + 128 negative items = 129 candidates
```

The model learns to assign a higher score to the true next item than to sampled negatives.

모델은 실제 다음 상품의 score가 sampled negative보다 높아지도록 학습합니다.

Sampled validation metrics are used only for monitoring. Final metrics are always computed against the complete catalog.

Sampled validation metric은 학습 상태 확인용으로만 사용하며, 최종 성능은 항상 전체 카탈로그를 대상으로 계산합니다.

---

## 7. Experiments | 실험

Five main experiments were evaluated under the same full-catalog protocol.

동일한 full-catalog 평가 조건에서 5개의 주요 실험을 비교했습니다.

| Experiment | Architecture | Recall@10 | Recall@20 | NDCG@20 | MRR@20 |
|---|---|---:|---:|---:|---:|
| A | 2-Layer Semantic SASRec | 0.336111 | 0.424044 | **0.193209** | **0.125545** |
| B | Semantic GRU | 0.028994 | 0.038031 | 0.017513 | 0.011612 |
| C | Semantic + Numeric SASRec | 0.196537 | 0.287215 | 0.121269 | 0.074911 |
| D | Semantic + Session Features SASRec | 0.309385 | 0.400546 | 0.182148 | 0.118841 |
| **E** | **4-Layer Semantic SASRec** | 0.335226 | **0.425167** | 0.193198 | 0.125286 |

Experiment E achieved the highest Recall@20, while Experiment A remained almost identical in ranking quality with lower architectural complexity.

Experiment E가 가장 높은 Recall@20을 기록했지만, Experiment A도 더 낮은 복잡도로 거의 동일한 ranking 성능을 보였습니다.

---

## 8. Evaluation Metrics | 평가 지표

The project reports:

프로젝트에서는 다음 지표를 사용합니다:

- Recall@1 / @5 / @10 / @20
- NDCG@10 / @20
- MRR@20

Final evaluation is performed against all `1,393,956` candidate products for `360,625` test sessions.

최종 평가는 `360,625`개의 테스트 세션에 대해 `1,393,956`개의 전체 후보 상품을 대상으로 수행합니다.

---

## 9. Inference Demo | 추론 데모

The final inference pipeline accepts a session of ASINs and a locale, converts ASINs to numerical IDs, generates a session representation and retrieves the highest-scoring products from the full catalog.

최종 inference pipeline은 ASIN 세션과 locale을 입력으로 받아 numerical ID로 변환하고, session representation을 만든 뒤 전체 카탈로그에서 가장 높은 score의 상품을 추천합니다.

Observed examples included:

실제 데모 결과 예시:

- a Top-1 hit for a labeling-product session;
- a Top-4 hit for a Jurassic World session;
- a heterogeneous-session failure case used for qualitative error analysis.

---

## 10. Repository Structure | 저장소 구조

```text
FinalProject/
├── README.md
├── TECHNICAL_REPORT.md
├── requirements.txt
├── .gitignore
├── configs/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── retrieval/
│   └── ranking/
├── artifacts/
│   ├── evaluation/
│   ├── demo/
│   ├── lookup/
│   └── mappings/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
└── tests/
```

Large raw datasets, model checkpoints and embedding matrices should not be committed directly to Git.

대용량 raw dataset, checkpoint 및 embedding matrix는 Git에 직접 커밋하지 않는 것을 권장합니다.

---

## 11. Installation | 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 12. Technical Report | 기술 보고서

For detailed EDA, preprocessing decisions, mathematical formulation, training strategy and all experiments, see:

상세 EDA, 전처리 결정, 수학적 정의, 학습 전략 및 전체 실험 내용은 다음 문서를 참고하세요:

[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

---

## 13. Conclusion | 결론

This project demonstrates a large-scale multilingual session-based recommendation pipeline combining frozen semantic item representations with Transformer-based sequential modeling.

본 프로젝트는 동결된 semantic item representation과 Transformer 기반 sequence modeling을 결합한 대규모 다국어 session-based recommendation pipeline을 구현했습니다.

The final 4-layer Semantic SASRec achieved a full-catalog Recall@20 of **0.425167** against approximately **1.39 million products**.

최종 4-layer Semantic SASRec은 약 **139만 개 상품**을 대상으로 **Recall@20 = 0.425167**을 기록했습니다.
