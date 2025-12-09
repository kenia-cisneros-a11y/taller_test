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

