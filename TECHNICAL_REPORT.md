# Technical Report: Amazon M2 Semantic Session-Based Recommender
# 기술 보고서: Amazon M2 의미 기반 세션 추천 시스템

## Abstract | 초록

This project develops and evaluates a large-scale multilingual session-based recommendation system using the Amazon M2 dataset. The system predicts the next product in a shopping session using frozen semantic product embeddings and sequential neural models. Product metadata is encoded with Voyage-4-Large into 1024-dimensional multilingual semantic vectors. Locale-specific product representations are aggregated into global product vectors, projected into a 128-dimensional latent space, and processed by SASRec-style Transformer encoders. Five architectures were compared under a full-catalog evaluation protocol over 1,393,956 products and 360,625 test sessions. The final 4-layer Semantic SASRec achieved Recall@20 = 0.425167, NDCG@20 = 0.193198 and MRR@20 = 0.125286.

본 프로젝트는 Amazon M2 데이터셋을 이용하여 대규모 다국어 세션 기반 추천 시스템을 개발하고 평가합니다. 시스템은 동결된 semantic product embedding과 sequential neural model을 이용해 쇼핑 세션의 다음 상품을 예측합니다. 상품 메타데이터는 Voyage-4-Large를 사용해 1024차원의 다국어 semantic vector로 변환됩니다. locale별 상품 표현은 global product vector로 통합되며, 128차원의 latent space로 projection한 뒤 SASRec 계열 Transformer encoder에서 처리됩니다. 총 5개의 아키텍처를 1,393,956개 상품과 360,625개 테스트 세션을 대상으로 full-catalog 방식으로 비교했습니다. 최종 4-layer Semantic SASRec은 Recall@20 = 0.425167, NDCG@20 = 0.193198, MRR@20 = 0.125286을 기록했습니다.

---

## 1. Problem Definition | 문제 정의

The recommendation task is next-item prediction from a session. A session is an ordered sequence of previously interacted products:

추천 문제는 세션 기반 next-item prediction입니다. 세션은 이전에 상호작용한 상품들의 순서가 있는 시퀀스입니다.

\[
S = [i_1, i_2, \ldots, i_T]
\]

The goal is to predict the next target product:

목표는 다음 target 상품을 예측하는 것입니다.

\[
i_{T+1}
\]

The system is session-based: no persistent user ID is required. The recommendation signal comes from the products observed in the current session and the session locale.

본 시스템은 session-based 방식으로 동작하며, 지속적인 user ID가 필요하지 않습니다. 추천 신호는 현재 세션에서 관찰된 상품들과 session locale에서 생성됩니다.

The production-style objective is retrieval from the full product catalog rather than classification among a small sampled set.

최종 목표는 작은 sampled candidate 집합에서의 분류가 아니라 전체 상품 카탈로그에서의 retrieval입니다.

---

## 2. Dataset | 데이터셋

### 2.1 Product Catalog | 상품 카탈로그

The product table contains 1,551,057 locale-specific metadata rows and 1,410,675 unique ASINs.

상품 테이블에는 1,551,057개의 locale-specific metadata row와 1,410,675개의 unique ASIN이 존재합니다.

Important columns include:

주요 컬럼:

```text
id
locale
title
price
brand
color
size
model
material
author
desc
```

The same ASIN may appear multiple times because the metadata can differ by locale.

동일한 ASIN은 locale별 메타데이터가 다를 수 있기 때문에 여러 row에 등장할 수 있습니다.

### 2.2 Session Data | 세션 데이터

The session table contains 3,606,249 rows with:

세션 테이블은 3,606,249개의 row로 구성됩니다.

```text
prev_items
next_item
locale
```

`prev_items` contains the ordered session history and `next_item` is the prediction target.

`prev_items`는 순서가 있는 세션 history이며 `next_item`은 예측 target입니다.

### 2.3 Locale Distribution | Locale 분포

Session counts are strongly imbalanced:

세션 수는 locale별로 불균형합니다.

| Locale | Sessions | Share |
|---|---:|---:|
| UK | 1,182,181 | 32.78% |
| DE | 1,111,416 | 30.82% |
| JP | 979,119 | 27.15% |
| IT | 126,925 | 3.52% |
| FR | 117,561 | 3.26% |
| ES | 89,047 | 2.47% |

This imbalance matters for splitting and evaluation. The original data order was not safe for a naive tail split because later rows could be dominated by a single locale.

이 불균형은 split과 evaluation에 직접적인 영향을 줍니다. 원본 데이터 순서는 단순한 tail split에 적합하지 않았으며, 뒤쪽 row가 특정 locale에 편중될 수 있었습니다.

---

## 3. Exploratory Data Analysis | 탐색적 데이터 분석

### 3.1 Product Distribution | 상품 분포

Product metadata is also highly concentrated in DE, UK and JP. This means model quality must be interpreted together with locale distribution.

상품 메타데이터 역시 DE, UK, JP에 크게 집중되어 있습니다. 따라서 모델 성능은 locale 분포와 함께 해석해야 합니다.

### 3.2 Price Anomaly | 가격 이상치

A repeated abnormal value was discovered:

반복적으로 등장하는 비정상 가격 값:

```text
40,000,000.07
```

This value occurred across unrelated products and locales and behaved like a systematic sentinel rather than a meaningful extreme price.

이 값은 서로 관련 없는 상품과 여러 locale에서 반복되어 실제 고가 상품보다 systematic sentinel에 가까운 패턴을 보였습니다.

Raw global price statistics were heavily distorted:

원본 전체 가격 통계는 심하게 왜곡되었습니다.

```text
Median: 23.28
Mean: ~1,806,107
Max: 40,000,000.07
```

Prices were therefore processed per locale. The abnormal sentinel was replaced by a locale-level central value before standardization, and StandardScaler was applied separately within each locale.

따라서 가격은 locale별로 처리했습니다. 비정상 sentinel은 locale 중심값으로 대체한 뒤 각 locale 내부에서 StandardScaler를 적용했습니다.

### 3.3 Missing Metadata | 메타데이터 결측치

Metadata fields have different missingness patterns. For example, `author` is missing for most products, while `brand`, `color`, `size`, `model`, `material` and `desc` have varying degrees of sparsity.

메타데이터 컬럼은 서로 다른 결측 패턴을 보입니다. `author`는 대부분의 상품에서 누락되며, `brand`, `color`, `size`, `model`, `material`, `desc` 역시 서로 다른 수준의 결측률을 가집니다.

Whitespace-only strings were normalized to missing values before downstream processing.

공백 문자열은 downstream processing 전에 missing value로 정규화했습니다.

### 3.4 Session Length | 세션 길이

Session lengths are short on average but have a long tail.

세션 길이는 평균적으로 짧지만 long-tail 분포를 가집니다.

Observed statistics:

```text
Mean: 4.24
Median: 3
90th percentile: 8
95th percentile: 11
99th percentile: 18
Max: 474
```

Because sequence lengths vary, dynamic padding is used at batch time instead of pre-padding every session to the global maximum.

시퀀스 길이가 서로 다르기 때문에 전체 최대 길이로 미리 padding하지 않고 batch 단계에서 dynamic padding을 사용합니다.

### 3.5 Duplicate Sessions | 중복 세션

117,357 exact duplicate sessions were identified in the original data. Duplicate patterns must be handled carefully because random row-level splitting can leak identical behavior across train, validation and test.

원본 데이터에서 117,357개의 exact duplicate session을 확인했습니다. 동일 패턴이 train, validation, test에 분산되면 leakage가 발생할 수 있으므로 주의가 필요합니다.

---

## 4. Preprocessing Pipeline | 전처리 파이프라인

The final preprocessing workflow was designed as a reusable multi-stage pipeline.

최종 전처리 workflow는 재사용 가능한 multi-stage pipeline으로 설계했습니다.

### 4.1 Empty-String Normalization | 공백 문자열 정규화

Object/string columns are stripped and whitespace-only values are converted to missing values.

문자열 컬럼을 strip한 후 공백만 존재하는 값을 missing value로 변환합니다.

This protects downstream text construction and categorical processing from silent empty-string errors.

이는 text construction과 categorical processing에서 공백 문자열이 정상 값처럼 처리되는 문제를 방지합니다.

### 4.2 Session Parsing | 세션 파싱

The raw session history is stored as a string representation such as:

원본 session history는 다음과 같은 문자열 형태입니다.

```text
['B09...' 'B08...' 'B07...']
```

A regex parser extracts ASINs into a true Python list.

정규식을 사용해 ASIN을 실제 Python list로 변환합니다.

### 4.3 Behavioral Vocabulary | 행동 기반 vocabulary

Unique ASINs from both `prev_items` and `next_item` are merged to determine all products that actually participate in behavioral learning.

`prev_items`와 `next_item`의 unique ASIN을 합쳐 실제 행동 학습에 참여하는 상품 집합을 구성합니다.

This produced 1,393,956 behavioral products.

최종 behavioral product 수는 1,393,956개입니다.

### 4.4 ASIN → Numerical ID Mapping | ASIN → 숫자 ID 매핑

Each behavioral ASIN receives one stable numerical ID.

각 behavioral ASIN에 하나의 고정 numerical ID를 부여합니다.

```text
0 = PAD
1 ... 1,393,956 = real items
```

The same ASIN keeps the same numerical ID across locales.

동일 ASIN은 locale이 달라도 동일한 numerical ID를 사용합니다.

### 4.5 Split Strategy | 데이터 분할 전략

Sessions are shuffled with a fixed random seed before splitting because the original order is locale-skewed.

원본 순서가 locale 편향을 가지기 때문에 fixed random seed로 shuffle한 후 split합니다.

Final split:

```text
Train:      2,884,999
Validation:   360,625
Test:         360,625
```

The project intentionally keeps repeated session patterns in training to maximize learning signal, while validation and test are built from the remaining sessions.

반복되는 session pattern은 학습 신호를 최대화하기 위해 train에 유지하고, 남은 세션으로 validation/test를 구성했습니다.

### 4.6 Locale Encoding | Locale 인코딩

Locale is categorical and has no ordinal relationship, so One-Hot Encoding is used instead of assigning arbitrary integer order.

Locale은 순서 관계가 없는 categorical variable이므로 임의의 정수 순서를 부여하지 않고 One-Hot Encoding을 사용합니다.

### 4.7 Price Processing | 가격 처리

Price is processed separately for each locale because currencies and price scales differ strongly between JP and European locales.

JP와 유럽 locale의 통화 및 가격 스케일이 크게 다르기 때문에 가격은 locale별로 처리합니다.

### 4.8 Model-Ready Samples | 모델 입력 샘플 생성

The final preprocessing stage separates:

최종 단계에서는 다음을 분리합니다.

- numerical catalog features;
- product text representations;
- embedding IDs;
- train/validation/test inputs;
- train/validation/test targets.

---

## 5. Semantic Product Representation | 상품 Semantic Representation

### 5.1 Product Text Construction | 상품 텍스트 구성

Available metadata fields are concatenated into structured product text:

존재하는 메타데이터 필드를 구조화된 상품 텍스트로 결합합니다.

```text
Title | ...
Brand | ...
Color | ...
Size | ...
Model | ...
Material | ...
Author | ...
Product Description | ...
```

Missing fields are omitted.

누락된 필드는 제외합니다.

### 5.2 Multilingual Encoder | 다국어 Encoder

Voyage-4-Large is used as the external semantic encoder.

외부 semantic encoder로 Voyage-4-Large를 사용합니다.

```text
Embedding dimension = 1024
```

Locale-specific product rows are embedded independently.

locale별 product row는 독립적으로 embedding합니다.

### 5.3 Multi-Locale Aggregation | Multi-Locale 통합

If ASIN `i` has `M_i` locale-specific representations, the global semantic representation is the mean:

ASIN `i`가 `M_i`개의 locale representation을 가진다면 global semantic representation은 다음과 같이 평균합니다.

\[
e_i = \frac{1}{M_i}\sum_{m=1}^{M_i} e_i^{(m)}
\]

The resulting frozen matrix is:

최종 frozen matrix:

```text
1,393,957 × 1024
```

Row 0 is reserved for padding.

0번 row는 padding 전용입니다.

---

## 6. Dataset and Batching | Dataset 및 배치 처리

A custom PyTorch `Dataset` returns three tensors for each sample:

Custom PyTorch `Dataset`은 각 sample에서 다음 세 tensor를 반환합니다.

```text
sequence: torch.long
locale:   torch.float32
 target:   torch.long
```

Because sessions have variable length, a custom `collate_fn` applies dynamic padding.

세션 길이가 서로 다르기 때문에 custom `collate_fn`에서 dynamic padding을 적용합니다.

For a batch size `B` and batch maximum sequence length `T`:

batch size가 `B`, 해당 batch의 최대 sequence length가 `T`일 때:

```text
sequences → [B, T]
locales   → [B, 6]
targets   → [B]
```

A padding mask prevents PAD positions from affecting self-attention.

padding mask를 사용해 PAD 위치가 self-attention에 영향을 주지 않도록 합니다.

---

## 7. Final Model Mathematics | 최종 모델 수학

### 7.1 Semantic Projection | Semantic Projection

Frozen 1024-dimensional product embeddings are projected into the model space:

동결된 1024차원 product embedding을 model space로 projection합니다.

\[
z_i = \operatorname{LayerNorm}(\operatorname{GELU}(W e_i + b))
\]

with:

```text
1024 → 128
```

The projected vector is L2-normalized before retrieval scoring.

retrieval scoring 전에 projected vector를 L2 normalize합니다.

### 7.2 Positional and Locale Information | Position 및 Locale 정보

For item position `t`:

\[
x_t = z_{i_t} + p_t + l
\]

where:

- `z_i` = projected semantic item representation;
- `p_t` = learnable positional embedding;
- `l` = projected locale vector.

여기서 `z_i`는 semantic item vector, `p_t`는 learnable positional embedding, `l`은 projected locale vector입니다.

### 7.3 Self-Attention | Self-Attention

For each Transformer layer:

\[
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
\]

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
\]

`M` contains the causal mask and padding constraints.

`M`에는 causal mask와 padding 제약이 포함됩니다.

### 7.4 Session Representation | 세션 Representation

After the Transformer, the representation of the last valid item is used as the session vector:

Transformer 이후 마지막 유효 item의 representation을 session vector로 사용합니다.

\[
h_s = H_{T_{valid}}
\]

followed by LayerNorm and L2 normalization.

이후 LayerNorm과 L2 normalization을 적용합니다.

### 7.5 Product Scoring | 상품 Scoring

For candidate product `j`:

\[
score(s,j)=h_s^T v_j
\]

Higher dot product means greater similarity between session intent and candidate product.

더 높은 dot product는 session intent와 candidate product의 더 높은 유사도를 의미합니다.

---

## 8. Training Strategy | 학습 전략

### 8.1 Negative Sampling | Negative Sampling

Computing scores against 1.39 million products for every training batch would be too expensive.

매 training batch마다 139만 개 상품 전체와 score를 계산하는 것은 계산 비용이 매우 큽니다.

Therefore each positive target is trained against 128 sampled negatives:

따라서 각 positive target에 대해 128개의 sampled negative를 사용합니다.

```text
1 positive + 128 negatives = 129 candidates
```

### 8.2 Training Objective | 학습 목적함수

Candidate scores are converted to a probability distribution:

candidate score를 확률 분포로 변환합니다.

\[
P(i|s)=\frac{\exp(score(s,i))}{\sum_j \exp(score(s,j))}
\]

The loss is cross entropy / negative log likelihood of the positive candidate:

loss는 positive candidate에 대한 cross entropy / negative log likelihood입니다.

\[
\mathcal{L}=-\log P(i^+|s)
\]

### 8.3 Optimization | 최적화

Main training settings:

주요 학습 설정:

```text
Optimizer: AdamW
Learning rate: 1e-3
Weight decay: 1e-4
Epochs: 3
Gradient clipping: 1.0
Negative samples: 128
```

---

## 9. Evaluation Protocol | 평가 프로토콜

### 9.1 Sampled Validation | Sampled Validation

During training, sampled validation is used only to monitor optimization progress.

학습 중 sampled validation은 optimization 진행 상태를 확인하기 위한 용도로만 사용합니다.

A sampled Recall@10 around 0.93 does not mean that the model retrieves the correct product from the full catalog with 93% probability, because the sampled task contains only 129 candidates.

Sampled Recall@10 약 0.93은 전체 카탈로그에서 93%의 확률로 정답을 찾는다는 의미가 아닙니다. sampled task에는 129개의 candidate만 존재하기 때문입니다.

### 9.2 Full-Catalog Evaluation | Full-Catalog 평가

Final evaluation compares the session representation against all 1,393,956 products.

최종 평가는 session representation을 전체 1,393,956개 상품과 비교합니다.

This is the primary evaluation protocol used for reported results.

보고된 최종 성능은 이 full-catalog protocol을 기준으로 합니다.

### 9.3 Recall@K | Recall@K

\[
Recall@K = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}(rank_n \le K)
\]

### 9.4 NDCG@K | NDCG@K

With one relevant next item per session:

세션당 하나의 relevant target이 있을 때:

\[
NDCG@K =
\begin{cases}
\frac{1}{\log_2(rank+1)}, & rank \le K \\
0, & otherwise
\end{cases}
\]

### 9.5 MRR@K | MRR@K

\[
MRR@K = \frac{1}{N}\sum_{n=1}^{N}
\begin{cases}
\frac{1}{rank_n}, & rank_n \le K \\
0, & otherwise
\end{cases}
\]

---

## 10. Experiments | 실험

### Experiment A — 2-Layer Semantic SASRec

**Hypothesis | 가설**  
Frozen semantic item representations combined with Transformer self-attention are sufficient to capture session intent.

동결된 semantic item representation과 Transformer self-attention의 결합만으로 session intent를 충분히 학습할 수 있다고 가정했습니다.

**Architecture | 아키텍처**

```text
1024-D frozen semantic embedding
→ 128-D projection
→ positional + locale conditioning
→ 2 Transformer layers
→ final session vector
→ dot-product retrieval
```

**Result | 결과**

```text
Recall@10 = 0.336111
Recall@20 = 0.424044
NDCG@20  = 0.193209
MRR@20   = 0.125545
```

**Interpretation | 해석**  
This became the strongest efficiency baseline and remained nearly identical to the deeper final model.

높은 효율성과 강한 성능을 가진 baseline이며, 더 깊은 최종 모델과 거의 동일한 ranking 품질을 보였습니다.

### Experiment B — Semantic GRU

**Hypothesis | 가설**  
A recurrent encoder may model sequence order with lower architectural complexity.

더 단순한 recurrent encoder로도 sequence order를 학습할 수 있는지 검증했습니다.

**Modification | 변경점**  
Transformer sequence encoder was replaced with a GRU-based sequence model.

Transformer sequence encoder를 GRU 기반 모델로 교체했습니다.

**Result | 결과**

```text
Recall@10 = 0.028994
Recall@20 = 0.038031
NDCG@20  = 0.017513
MRR@20   = 0.011612
```

**Interpretation | 해석**  
GRU performance dropped sharply under full-catalog retrieval. Self-attention was substantially better for this setup.

full-catalog retrieval에서 GRU 성능이 크게 감소했으며, 본 문제에서는 self-attention이 훨씬 효과적이었습니다.

### Experiment C — Semantic + Numeric SASRec

**Hypothesis | 가설**  
Price and locale metadata may provide complementary product signals beyond text semantics.

가격과 locale 정보가 text semantic 이외의 추가 product signal을 제공할 수 있다고 가정했습니다.

**Modification | 변경점**  
A numerical branch was added using standardized price and locale OHE and fused with semantic item representations.

표준화된 price와 locale OHE를 사용하는 numerical branch를 추가하고 semantic item representation과 fusion했습니다.

**Result | 결과**

```text
Recall@10 = 0.196537
Recall@20 = 0.287215
NDCG@20  = 0.121269
MRR@20   = 0.074911
```

**Interpretation | 해석**  
The chosen fusion strategy degraded performance. Locale-specific numerical metadata does not combine cleanly with a single global product representation.

선택한 fusion 방식은 성능을 저하시켰습니다. locale-specific numerical metadata는 global product representation과 단순하게 결합하기 어렵다는 점을 확인했습니다.

### Experiment D — Semantic + Session Features SASRec

**Hypothesis | 가설**  
Explicit session statistics may improve sequence representations.

명시적인 session statistics가 sequence representation을 보완할 수 있다고 가정했습니다.

**Additional features | 추가 feature**

```text
session length
unique item count
unique / length ratio
```

**Result | 결과**

```text
Recall@10 = 0.309385
Recall@20 = 0.400546
NDCG@20  = 0.182148
MRR@20   = 0.118841
```

**Interpretation | 해석**  
The model remained strong but did not surpass the semantic-only baseline. Much of this information may already be implicitly represented by Transformer attention and repeated-item patterns.

성능은 여전히 강했지만 semantic-only baseline을 넘지 못했습니다. 해당 정보의 상당 부분이 Transformer attention과 반복 item 패턴에 이미 암묵적으로 포함되어 있을 가능성이 있습니다.

### Experiment E — 4-Layer Semantic SASRec

**Hypothesis | 가설**  
Additional Transformer depth may capture more complex item-to-item transitions.

더 깊은 Transformer가 복잡한 item transition을 더 잘 학습할 수 있는지 검증했습니다.

**Modification | 변경점**

```text
Transformer layers: 2 → 4
```

Other core settings remained unchanged.

다른 핵심 설정은 동일하게 유지했습니다.

**Result | 결과**

```text
Recall@10 = 0.335226
Recall@20 = 0.425167
NDCG@20  = 0.193198
MRR@20   = 0.125286
```

**Interpretation | 해석**  
This model achieved the highest Recall@20, but the improvement over the 2-layer baseline was marginal. The result demonstrates diminishing returns from additional depth.

가장 높은 Recall@20을 기록했지만 2-layer baseline 대비 개선 폭은 매우 작았습니다. 이는 model depth 증가에 따른 diminishing returns를 보여줍니다.

---

## 11. Experimental Results | 실험 결과

| Experiment | Architecture | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@20 | MRR@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 2-Layer Semantic SASRec | 0.028448 | **0.243339** | **0.336111** | 0.424044 | **0.193209** | **0.125545** |
| B | Semantic GRU | 0.004140 | 0.020229 | 0.028994 | 0.038031 | 0.017513 | 0.011612 |
| C | Semantic + Numeric SASRec | 0.026274 | 0.123283 | 0.196537 | 0.287215 | 0.121269 | 0.074911 |
| D | Semantic + Session Features | **0.034099** | 0.218498 | 0.309385 | 0.400546 | 0.182148 | 0.118841 |
| **E** | **4-Layer Semantic SASRec** | 0.028415 | 0.242487 | 0.335226 | **0.425167** | 0.193198 | 0.125286 |

The final model is selected primarily by Recall@20 because the project focuses on candidate retrieval quality. However, the 2-layer model is nearly tied and provides a better complexity-efficiency trade-off.

프로젝트가 candidate retrieval 성능에 초점을 두기 때문에 최종 모델은 Recall@20을 우선 기준으로 선택했습니다. 다만 2-layer 모델도 거의 동일한 성능을 보여 complexity-efficiency trade-off 측면에서는 매우 경쟁력이 있습니다.

---

## 12. Qualitative Inference Analysis | 정성적 추론 분석

### 12.1 Successful Labeling Session | 라벨링 상품 세션 성공 사례

A session containing DYMO and cable-labeling products produced the real next product at rank 1.

DYMO 및 cable-labeling 상품으로 구성된 세션에서 실제 다음 상품이 rank 1에 등장했습니다.

This demonstrates that the model can map a coherent session into a meaningful local product neighborhood.

이는 coherent session을 의미 있는 local product neighborhood로 매핑할 수 있음을 보여줍니다.

### 12.2 Jurassic World Session | Jurassic World 세션

A session containing several Jurassic World editions produced a recommendation list dominated by semantically related editions. The true next product appeared at rank 4.

여러 Jurassic World 상품이 포함된 세션에서는 유사한 영화 edition이 추천 목록 상위를 차지했으며 실제 next item은 rank 4에 위치했습니다.

### 12.3 Failure Case | 실패 사례

A heterogeneous session containing tablets, cosmetics, Christmas products, cooking items and spices failed to retrieve the true Barbie target in Top-10.

태블릿, 화장품, 크리스마스 상품, 요리 관련 상품, 향신료가 혼합된 heterogeneous session에서는 실제 Barbie target을 Top-10에 포함하지 못했습니다.

The recommendations concentrated on the most recent cooking/spice intent. This suggests that multi-interest session modeling could improve future versions.

추천 결과는 최근의 cooking/spice intent에 집중되었습니다. 향후 multi-interest session modeling이 개선 방향이 될 수 있습니다.

---

## 13. Inference Pipeline | 추론 파이프라인

The final runtime pipeline is:

최종 runtime pipeline:

```text
ASIN session
→ ASIN-to-num_id mapping
→ frozen global embeddings
→ 4-layer SASRec session encoder
→ 128-D session vector
→ dot product against projected catalog vectors
→ Top-K products
→ metadata lookup
```

Projected catalog vectors are precomputed and stored so the 1024→128 product projection is not repeated for every request.

1024→128 product projection을 매 요청마다 반복하지 않도록 projected catalog vector를 사전 계산하여 저장합니다.

---

## 14. Limitations | 한계점

1. Full-catalog scoring is brute-force and does not yet use FAISS or another ANN index.  
   Full-catalog scoring은 brute-force 방식이며 아직 FAISS 등의 ANN index를 사용하지 않습니다.

2. Locale is used as context, but final retrieval is global and does not strictly filter availability by locale.  
   Locale을 context로 사용하지만 최종 retrieval은 global이며 locale availability를 강제 filter하지 않습니다.

3. Semantic embeddings are frozen.  
   Semantic embedding은 frozen 상태입니다.

4. Random negative sampling can be improved with hard-negative mining.  
   Random negative sampling은 hard-negative mining으로 개선할 수 있습니다.

5. Heterogeneous sessions may contain multiple simultaneous intents.  
   Heterogeneous session은 여러 intent를 동시에 포함할 수 있습니다.

---

## 15. Future Work | 향후 개선

Potential extensions include:

향후 개선 방향:

```text
FAISS / ANN candidate retrieval
hard-negative mining
locale-specific availability filtering
multi-interest session encoders
learnable semantic aggregation
cross-encoder reranking
online inference API
```

A future production architecture could use a two-stage system:

향후 production 구조는 two-stage system으로 확장할 수 있습니다.

```text
Session Encoder
    ↓
ANN Retrieval (Top 100–500)
    ↓
Learned Ranker / Cross-Encoder
    ↓
Final Top-K
```

---

## 16. Conclusion | 결론

This project demonstrates a complete large-scale multilingual session-recommendation workflow, including EDA, leakage-aware preprocessing, multilingual semantic representation, sequential modeling, negative-sampled training, full-catalog evaluation and qualitative inference analysis.

본 프로젝트는 EDA, leakage-aware preprocessing, 다국어 semantic representation, sequential modeling, negative-sampled training, full-catalog evaluation, 정성적 inference analysis까지 포함하는 완전한 대규모 다국어 session recommendation workflow를 구현했습니다.

The strongest full-catalog retrieval result was achieved by the 4-layer Semantic SASRec:

가장 높은 full-catalog retrieval 성능은 4-layer Semantic SASRec에서 기록했습니다.

```text
Recall@20 = 0.425167
NDCG@20  = 0.193198
MRR@20   = 0.125286
```

The project also shows that additional feature complexity does not automatically improve recommendation quality. Controlled experiments with GRU, numeric metadata, session-level features and additional Transformer depth provided clear evidence for selecting the final architecture.

또한 추가 feature와 architecture complexity가 항상 성능 향상으로 이어지지 않는다는 점을 확인했습니다. GRU, numerical metadata, session-level feature, 추가 Transformer depth에 대한 controlled experiment를 통해 최종 architecture 선택 근거를 확보했습니다.
