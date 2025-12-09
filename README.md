# Technical challenge for Taller Technologies

## Business Context
- **Problem**: 22% of products are returned, costing $18 per return (~$400K/month)
- **Opportunity**: Predict returns and apply ($3 per order) interventions that reduce return probability by 35%
- **Goal**: Identify high-risk orders for targeted intervention

## Critical Insight from Baseline Model
The baseline model has **0% recall on returns** - it predicts ALL orders as "kept".

This is completely useless for our business case: we need to CATCH returns to intervene.

The baseline results show a **catastrophic failure**:
- Recall on returns = 0% (catches NO returns)
- The model simply predicts "kept" for everything
- This happens due to class imbalance (78% kept vs 22% returned)

From the bussiness impact we can see is expensiest use the actual model comparign without the model.

## Metric Selection Justification

| Metric | Why It Matters | Priority | Target |
|--------|---------------|----------|--------|
| **Recall (Returns)** | Each missed return costs $18. We MUST catch returns. | HIGH | > 50% |
| **Precision (Returns)** | Unnecessary interventions cost $3. Less critical due to cost asymmetry. | MEDIUM | > 50% |
| **PR-AUC** | Better than ROC-AUC for imbalanced data, focuses on minority class | HIGH | > 0.40 |
| **Business Savings** | Ultimate measure of model value | CRITICAL | > $0 |


With these metrics, we prioritize catching returns, minimizing unnecessary interventions, and maximizing savings knowing that we need a model with a minimum precision for profitability: **47.6%** (Good enough to deploy).

## Model Strengths & Weakness Identification


    *   Strengths

High specificity for non-returns — The model correctly identifies 1,405 out of 1,495 "kept" items (94% specificity), minimizing unnecessary interventions on customers who wouldn't return anyway.
Precision above random — At 29.13% precision, the model is ~2x better than random guessing (base return rate ~25%), meaning predictions aren't entirely noise.
Low false positive cost — Only $270 spent on unnecessary interventions (90 × $3), so when it does flag something, the downside risk is contained.


    *   Weaknesses

Catastrophic recall failure — Only 7.33% recall means 468 of 505 returns go undetected, costing $8,424 in missed opportunities.
Net negative ROI — The model loses $147.90 compared to doing nothing. It's currently worse than not having a model at all.
Below profitability threshold — Minimum precision needed is 47.6%; the model achieves only 29.13%.


    *   Where Does It Fail Most?
Fashion category is the biggest blind spot:

Highest return rate (31.3%) and largest volume (346 returns)
Worst recall at just 6.07% — catching only ~21 of 346 returns
Represents $1,141.80 in potential savings left on the table — more than Electronics and Home_Decor combined

The model systematically misses the highest-risk, highest-volume segment.

    *   Is Accuracy the Right Metric?
No — accuracy is misleading here. Here's why:
IssueExplanationClass imbalance~75% of items are kept. A model predicting "kept" for everything would achieve ~75% accuracy while catching zero returns.Asymmetric costsMissing a return costs 6× more than a false alarm ($18 vs $3). Accuracy treats all errors equally.Business misalignmentThe goal is cost reduction, not correct classifications. A model with lower accuracy but higher recall could save significantly more money.

## Recommendations for Model Improvement

1. Use a model for inbalanced datasets for training and validation to improve model performance.
2. Use apropiate scoring metric like recall.

## Model Improvement

**Problem:** Current model loses $147.90 per period. It catches only 37 of 505 returns (7.33%) while 468 go undetected, costing $8,424 in missed interventions.

**Opportunity:** Fashion category alone represents $1,141.80 in recoverable savings. Total addressable savings across all categories: $1,667.

**Target:** Achieve 50-60% recall while maintaining precision above the 47.6% profitability threshold. Expected savings: $600-$1,000 per period.

**ROI:** Model improvement effort pays back within first evaluation period.

    *   Evaluation Metrics (Beyond Accuracy)

Accuracy is misleading for this problem. With 75% non-returns, a model predicting "kept" for everything achieves 75% accuracy but zero business value.

| Metric | Why It Matters | Priority | Target |
|--------|---------------|----------|--------|
| **Recall (Returns)** | Each missed return costs $18. We MUST catch returns. | HIGH | > 50% |
| **Precision (Returns)** | Unnecessary interventions cost $3. Less critical due to cost asymmetry. | MEDIUM | > 50% |
| **PR-AUC** | Better than ROC-AUC for imbalanced data, focuses on minority class | HIGH | > 0.40 |
| **Business Savings** | Ultimate measure of model value | CRITICAL | > $0 |

    *   Class Imbalance Analysis

**Distribution:** 75% kept vs 25% returned (3:1 ratio)

**Impact:** Model biases toward majority class, producing high specificity but near-zero recall.

**Mitigation strategies to test:**

| Technique | Implementation | Risk |
|-----------|----------------|------|
| class_weight='balanced' | Built-in sklearn parameter | None |
| SMOTE | Apply to training data only | Leakage if applied to test |
| Threshold adjustment | Lower from 0.5 to 0.25-0.35 | May hurt precision |
| Cost-sensitive weights | Weight FN at 6x FP cost | Requires custom loss |

    *    Cost-Benefit Framework

| Event | Cost/Benefit | Frequency (Current) |
|-------|--------------|---------------------|
| True Positive (caught return) | +$3.30 saved | 37 |
| False Positive (unnecessary intervention) | -$3.00 wasted | 90 |
| False Negative (missed return) | -$18.00 lost | 468 |
| True Negative (correct no-action) | $0.00 | 1,405 |

**Break-even precision:** $3 / ($3 + $3.30) = 47.6%

**Decision rule:** Accept any model configuration where precision exceeds 47.6% and recall is maximized.

    *    Baseline Comparison Framework

All improvements measured against documented baseline:

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 |
|--------|----------|---------|---------|---------|
| Model | LogReg (default) | LogReg (tuned threshold) | LogReg + features | Best algorithm |
| Recall | 7.33% | -- | -- | -- |
| Precision | 29.13% | -- | -- | -- |
| Savings | -$147.90 | -- | -- | -- |

**Requirement:** Each phase must show statistically significant improvement (p < 0.05) via paired t-test on cross-validation folds.

    *   Validation Strategy

**Data splits:**
- Train: 60% (stratified by target and category)
- Validation: 20% (hyperparameter tuning)
- Test: 20% (final evaluation, touched once)

**Cross-validation:** 5-fold stratified CV for all experiments. Report mean and standard deviation.

**Leakage prevention checklist:**
- All feature engineering fitted on train only
- Category averages calculated on train only
- SMOTE applied to train only
- Threshold optimized on validation, evaluated on test

---

    *   Feature Engineering (With Rationale)

| Feature | Hypothesis | Business Logic | Leakage Check |
|---------|------------|----------------|---------------|
| return_rate = previous_returns / tenure_days | Historical behavior predicts future | Repeat returners are identifiable | Safe: uses pre-order data |
| is_extreme_size | XS/XXL have fit uncertainty | Size charts fail at extremes | Safe: known at order time |
| price_vs_category_avg | Expensive items face scrutiny | Higher expectations, more returns | Calculate mean on train only |
| is_new_customer | Tenure < 30 days | New customers lack trust | Safe: known at order time |
| discount_x_price | Interaction term | Impulse buys on big discounts | Safe: known at order time |
| low_rating_flag | Rating < 3.5 | Poor products return more | Safe: pre-existing rating |

**Excluded features:**
- order_id: No predictive value, only identifier
- Any post-purchase data: Would cause leakage



    *   Overfitting Prevention

| Technique | Implementation |
|-----------|----------------|
| Train/test gap monitoring | Flag if test recall > 10% below train recall |
| Cross-validation | 5-fold CV to ensure stability |
| Regularization | L1/L2 penalty in logistic regression, tune C parameter |
| Tree constraints | max_depth, min_samples_leaf in RF/GBM |
| Early stopping | For gradient boosting methods |
| Learning curves | Plot train vs validation performance across data sizes |

**Acceptance criteria:** Model passes if CV standard deviation < 5% of mean for primary metrics.

    *   Deployment Plan

**Phase 1: Shadow Mode (Week 1-2)**
- Run model in parallel with current process
- Log predictions without acting on them
- Compare predicted vs actual returns
- Validate precision remains above 47.6%

**Phase 2: Limited Rollout (Week 3-4)**
- Deploy to Electronics category only (lowest risk)
- Intervene on flagged orders
- Measure actual savings vs predicted

**Phase 3: Full Deployment (Week 5+)**
- Extend to Fashion and Home_Decor
- Implement category-specific thresholds if needed

**Rollback trigger:** If precision drops below 45% on rolling 7-day window, revert to previous version.

    *   Data Drift Monitoring

**Features to monitor:**

| Feature | Expected Range | Alert Threshold |
|---------|----------------|-----------------|
| Return rate | 20-30% | Outside 15-35% |
| Category distribution | Fashion ~55% | Shift > 10pp |
| Average price | $30-50 | Change > 20% |
| New customer ratio | ~15% | Change > 5pp |

**Model performance monitoring:**
- Weekly precision/recall calculation
- Monthly full model evaluation
- Quarterly retraining assessment

**Retraining triggers:**
- Precision drops below 47.6% for 2 consecutive weeks
- Feature distributions shift beyond alert thresholds
- Business costs change (intervention cost, return cost)