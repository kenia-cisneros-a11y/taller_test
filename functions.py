import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    average_precision_score
)

def preprocess(df):
    """Simple preprocessing pipeline"""
    df_processed = df.copy()
    
    # Encode categorical: product_category
    le_category = LabelEncoder()
    df_processed['product_category_encoded'] = le_category.fit_transform(
        df_processed['product_category']
    )
    
    # Handle missing sizes (Fashion items only have sizes)
    if df_processed['size_purchased'].notna().any():
        most_common_size = df_processed['size_purchased'].mode()[0]
        df_processed['size_purchased'].fillna(most_common_size, inplace=True)
        
        le_size = LabelEncoder()
        df_processed['size_encoded'] = le_size.fit_transform(
            df_processed['size_purchased']
        )
    
    # Feature selection
    feature_cols = [
        'customer_age', 'customer_tenure_days', 'product_category_encoded',
        'product_price', 'days_since_last_purchase', 'previous_returns',
        'product_rating', 'size_encoded', 'discount_applied'
    ]
    
    X = df_processed[feature_cols]
    y = df_processed['is_return']
    
    return X, y

RETURN_COST = 18              # Cost per return ($)
INTERVENTION_COST = 3         # Cost per intervention ($)
INTERVENTION_EFFECT = 0.35    # Reduction in return probability (35%)
MONTHLY_ORDERS = 400000 / (0.22 * 18)  # Estimated monthly orders (~101K)

def calculate_return_rates_by_category(df):
    """
    Calculates the return rate for each product category.

    Parameters:
    df (DataFrame): Input dataset containing 'product_category' and 'is_return' columns.

    Returns:
    DataFrame: Return rates per category sorted in descending order.
    """
    # Group by product category and calculate mean of is_return (1 = returned, 0 = kept)
    return_rates = df.groupby('product_category')['is_return'].mean().reset_index()

    # Rename column for clarity
    return_rates.rename(columns={'is_return': 'return_rate'}, inplace=True)

    # Sort by return rate descending
    return_rates.sort_values(by='return_rate', ascending=False, inplace=True)

    return return_rates
def calculate_business_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate business-relevant metrics for return prediction.
    
    Key metrics:
    - Recall (Returns): % of actual returns we catch - MOST IMPORTANT
    - Precision (Returns): % of predicted returns that are actual returns
    - Business savings: Money saved by intervening on predicted returns
    """
    
    # Basic classification metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Core metrics
    recall_returns = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_returns = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_returns = 2 * (precision_returns * recall_returns) / (precision_returns + recall_returns) if (precision_returns + recall_returns) > 0 else 0
    
    # Business cost calculations
    # Without model: all returns cost full $18
    cost_no_model = (tp + fn) * RETURN_COST
    
    # With model intervention:
    # - TP (caught returns): pay intervention ($3), return still happens but 35% less likely
    #   Expected cost = $3 + $18 * (1 - 0.35) = $3 + $11.70 = $14.70
    # - FP (false alarms): pay intervention only = $3
    # - FN (missed returns): full return cost = $18
    # - TN (correct kept): no cost = $0
    
    cost_per_tp = INTERVENTION_COST + RETURN_COST * (1 - INTERVENTION_EFFECT)
    cost_per_fp = INTERVENTION_COST
    cost_per_fn = RETURN_COST
    
    cost_with_model = tp * cost_per_tp + fp * cost_per_fp + fn * cost_per_fn
    
    savings = cost_no_model - cost_with_model
    savings_per_order = savings / len(y_true) if len(y_true) > 0 else 0
    
    metrics = {
        'Recall (Returns)': recall_returns,
        'Precision (Returns)': precision_returns,
        'F1-Score (Returns)': f1_returns,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn,
        'Cost Without Model ($)': cost_no_model,
        'Cost With Model ($)': cost_with_model,
        'Total Savings ($)': savings,
        'Savings per Order ($)': savings_per_order
    }
    
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
        metrics['PR-AUC'] = average_precision_score(y_true, y_prob)
    
    return metrics


def plot_confusion_matrix_detailed(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix with business cost interpretation.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Standard confusion matrix
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Predicted: Kept', 'Predicted: Return'],
                yticklabels=['Actual: Kept', 'Actual: Return'])
    ax1.set_title(f'{title}\n(Counts)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    
    # Plot 2: Business cost interpretation
    ax2 = axes[1]
    tn, fp, fn, tp = cm.ravel()
    
    cost_per_tp = INTERVENTION_COST + RETURN_COST * (1 - INTERVENTION_EFFECT)
    
    total_cost_matrix = np.array([
        [0, fp * INTERVENTION_COST],
        [fn * RETURN_COST, tp * cost_per_tp]
    ])
    
    labels = np.array([
        [f'TN: {tn}\n$0 each\nTotal: $0', 
         f'FP: {fp}\n${INTERVENTION_COST} each\nTotal: ${fp * INTERVENTION_COST:,.0f}'],
        [f'FN: {fn}\n${RETURN_COST} each\nTotal: ${fn * RETURN_COST:,.0f}', 
         f'TP: {tp}\n${cost_per_tp:.1f} each\nTotal: ${tp * cost_per_tp:,.0f}']
    ])
    
    sns.heatmap(total_cost_matrix, annot=labels, fmt='', cmap='RdYlGn_r', ax=ax2,
                xticklabels=['Predicted: Kept', 'Predicted: Return'],
                yticklabels=['Actual: Kept', 'Actual: Return'])
    ax2.set_title(f'{title}\n(Business Cost Analysis)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Print interpretation
    print("\n📋 CONFUSION MATRIX INTERPRETATION:")
    print("-" * 50)
    print(f"• True Negatives (TN):  {tn:,} - Correctly predicted as kept (no cost)")
    print(f"• False Positives (FP): {fp:,} - Wrongly flagged for intervention (${INTERVENTION_COST}/each)")
    print(f"• False Negatives (FN): {fn:,} - MISSED RETURNS (${RETURN_COST}/each) ← BIGGEST PROBLEM")
    print(f"• True Positives (TP):  {tp:,} - Caught returns (reduced cost)")

def analyze_by_category(df, y_true_col, y_pred_col, category_col='product_category'):
    """
    Analyze model performance by product category.
    """
    categories = df[category_col].unique()
    
    results = []
    for cat in categories:
        mask = df[category_col] == cat
        y_true_cat = df.loc[mask, y_true_col]
        y_pred_cat = df.loc[mask, y_pred_col]
        
        n_samples = len(y_true_cat)
        n_returns = y_true_cat.sum()
        return_rate = n_returns / n_samples if n_samples > 0 else 0
        
        if n_returns > 0 and y_pred_cat.sum() > 0:
            recall = recall_score(y_true_cat, y_pred_cat)
            precision = precision_score(y_true_cat, y_pred_cat)
            f1 = f1_score(y_true_cat, y_pred_cat)
        else:
            recall = 0
            precision = 0
            f1 = 0
        
        # Business impact
        potential_savings = n_returns * RETURN_COST * INTERVENTION_EFFECT - n_returns * INTERVENTION_COST
        
        results.append({
            'Category': cat,
            'Samples': n_samples,
            'Returns': n_returns,
            'Return Rate': return_rate,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1,
            'Potential Savings ($)': potential_savings
        })
    
    return pd.DataFrame(results)