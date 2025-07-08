# COMPAS Recidivism Dataset Bias Audit
# Using IBM AI Fairness 360 Toolkit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# AI Fairness 360 imports
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class COMPASBiasAuditor:
    def __init__(self):
        self.dataset = None
        self.privileged_groups = [{'race': 1}]  # Caucasian
        self.unprivileged_groups = [{'race': 0}]  # African-American
        
    def load_and_prepare_data(self):
        """Load COMPAS dataset and prepare for analysis"""
        print("Loading COMPAS dataset...")
        
        # Load dataset using AIF360
        self.dataset = CompasDataset()
        
        # Convert to pandas DataFrame for easier manipulation
        df = self.dataset.convert_to_dataframe()[0]
        
        print(f"Dataset shape: {df.shape}")
        print("\nDataset columns:")
        print(df.columns.tolist())
        
        # Basic statistics
        print(f"\nRace distribution:")
        print(df['race'].value_counts())
        
        print(f"\nRecidivism rate by race:")
        print(df.groupby('race')['two_year_recid'].mean())
        
        return df
    
    def analyze_dataset_bias(self):
        """Analyze bias in the original dataset"""
        print("\n" + "="*50)
        print("DATASET BIAS ANALYSIS")
        print("="*50)
        
        # Calculate bias metrics
        metric = BinaryLabelDatasetMetric(
            self.dataset,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups
        )
        
        print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
        print(f"Disparate Impact: {metric.disparate_impact():.4f}")
        print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
        
        return metric
    
    def train_baseline_model(self):
        """Train a baseline model to evaluate prediction bias"""
        print("\n" + "="*50)
        print("BASELINE MODEL TRAINING")
        print("="*50)
        
        # Split data
        dataset_train, dataset_test = self.dataset.split([0.7], shuffle=True, seed=42)
        
        # Convert to numpy arrays
        X_train = dataset_train.features
        y_train = dataset_train.labels.ravel()
        X_test = dataset_test.features
        y_test = dataset_test.labels.ravel()
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Create dataset with predictions
        dataset_pred = dataset_test.copy()
        dataset_pred.labels = y_pred
        
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        return dataset_test, dataset_pred, rf_model
    
    def analyze_prediction_bias(self, dataset_test, dataset_pred):
        """Analyze bias in model predictions"""
        print("\n" + "="*50)
        print("PREDICTION BIAS ANALYSIS")
        print("="*50)
        
        # Calculate classification metrics
        metric = ClassificationMetric(
            dataset_test,
            dataset_pred,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups
        )
        
        print(f"Equalized Odds Difference: {metric.equalized_odds_difference():.4f}")
        print(f"Average Odds Difference: {metric.average_odds_difference():.4f}")
        print(f"Disparate Impact: {metric.disparate_impact():.4f}")
        
        # False Positive Rate Disparity
        fpr_diff = metric.false_positive_rate_difference()
        print(f"False Positive Rate Difference: {fpr_diff:.4f}")
        
        # False Negative Rate Disparity
        fnr_diff = metric.false_negative_rate_difference()
        print(f"False Negative Rate Difference: {fnr_diff:.4f}")
        
        return metric
    
    def create_visualizations(self, dataset_test, dataset_pred):
        """Generate comprehensive bias visualizations"""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Convert datasets to DataFrames
        df_test = dataset_test.convert_to_dataframe()[0]
        df_pred = dataset_pred.convert_to_dataframe()[0]
        
        # Merge actual and predicted labels
        df_combined = df_test.copy()
        df_combined['predicted'] = df_pred['two_year_recid']
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('COMPAS Recidivism Dataset Bias Analysis', fontsize=16, fontweight='bold')
        
        # 1. Recidivism Rate by Race
        recid_by_race = df_combined.groupby('race')['two_year_recid'].mean()
        axes[0, 0].bar(['African-American', 'Caucasian'], recid_by_race.values, 
                      color=['#ff7f0e', '#1f77b4'])
        axes[0, 0].set_title('Actual Recidivism Rate by Race')
        axes[0, 0].set_ylabel('Recidivism Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(recid_by_race.values):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. Predicted Recidivism Rate by Race
        pred_by_race = df_combined.groupby('race')['predicted'].mean()
        axes[0, 1].bar(['African-American', 'Caucasian'], pred_by_race.values,
                      color=['#ff7f0e', '#1f77b4'])
        axes[0, 1].set_title('Predicted Recidivism Rate by Race')
        axes[0, 1].set_ylabel('Predicted Recidivism Rate')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(pred_by_race.values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 3. False Positive Rates by Race
        fpr_by_race = []
        for race in [0, 1]:  # African-American, Caucasian
            race_data = df_combined[df_combined['race'] == race]
            actual_negative = race_data[race_data['two_year_recid'] == 0]
            if len(actual_negative) > 0:
                fpr = (actual_negative['predicted'] == 1).sum() / len(actual_negative)
                fpr_by_race.append(fpr)
            else:
                fpr_by_race.append(0)
        
        axes[0, 2].bar(['African-American', 'Caucasian'], fpr_by_race,
                      color=['#ff7f0e', '#1f77b4'])
        axes[0, 2].set_title('False Positive Rate by Race')
        axes[0, 2].set_ylabel('False Positive Rate')
        axes[0, 2].set_ylim(0, max(fpr_by_race) * 1.2)
        
        # Add value labels on bars
        for i, v in enumerate(fpr_by_race):
            axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 4. False Negative Rates by Race
        fnr_by_race = []
        for race in [0, 1]:  # African-American, Caucasian
            race_data = df_combined[df_combined['race'] == race]
            actual_positive = race_data[race_data['two_year_recid'] == 1]
            if len(actual_positive) > 0:
                fnr = (actual_positive['predicted'] == 0).sum() / len(actual_positive)
                fnr_by_race.append(fnr)
            else:
                fnr_by_race.append(0)
        
        axes[1, 0].bar(['African-American', 'Caucasian'], fnr_by_race,
                      color=['#ff7f0e', '#1f77b4'])
        axes[1, 0].set_title('False Negative Rate by Race')
        axes[1, 0].set_ylabel('False Negative Rate')
        axes[1, 0].set_ylim(0, max(fnr_by_race) * 1.2 if max(fnr_by_race) > 0 else 1)
        
        # Add value labels on bars
        for i, v in enumerate(fnr_by_race):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 5. Confusion Matrix Heatmap for African-American
        aa_data = df_combined[df_combined['race'] == 0]
        cm_aa = confusion_matrix(aa_data['two_year_recid'], aa_data['predicted'])
        sns.heatmap(cm_aa, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Recidivism', 'Recidivism'],
                   yticklabels=['No Recidivism', 'Recidivism'], ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix - African-American')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # 6. Confusion Matrix Heatmap for Caucasian
        c_data = df_combined[df_combined['race'] == 1]
        cm_c = confusion_matrix(c_data['two_year_recid'], c_data['predicted'])
        sns.heatmap(cm_c, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=['No Recidivism', 'Recidivism'],
                   yticklabels=['No Recidivism', 'Recidivism'], ax=axes[1, 2])
        axes[1, 2].set_title('Confusion Matrix - Caucasian')
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization: Bias metrics comparison
        self.create_bias_metrics_chart(dataset_test, dataset_pred)
    
    def create_bias_metrics_chart(self, dataset_test, dataset_pred):
        """Create a chart comparing various bias metrics"""
        metric = ClassificationMetric(
            dataset_test,
            dataset_pred,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups
        )
        
        # Collect bias metrics
        metrics = {
            'Statistical Parity\nDifference': metric.statistical_parity_difference(),
            'Disparate Impact': metric.disparate_impact() - 1,  # Normalized around 0
            'Equalized Odds\nDifference': metric.equalized_odds_difference(),
            'Average Odds\nDifference': metric.average_odds_difference(),
            'False Positive Rate\nDifference': metric.false_positive_rate_difference(),
            'False Negative Rate\nDifference': metric.false_negative_rate_difference()
        }
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = ['red' if v > 0.1 or v < -0.1 else 'orange' if abs(v) > 0.05 else 'green' 
                 for v in metric_values]
        
        bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(value + 0.01 if value >= 0 else value - 0.01, i, 
                   f'{value:.3f}', va='center', 
                   ha='left' if value >= 0 else 'right', fontweight='bold')
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='High Bias Threshold')
        ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, label='Moderate Bias Threshold')
        ax.axvline(x=-0.05, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Bias Metric Value')
        ax.set_title('Bias Metrics Summary - COMPAS Dataset\n(Negative values favor African-American, Positive favor Caucasian)')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def apply_bias_mitigation(self):
        """Apply bias mitigation techniques"""
        print("\n" + "="*50)
        print("BIAS MITIGATION")
        print("="*50)
        
        # Split data
        dataset_train, dataset_test = self.dataset.split([0.7], shuffle=True, seed=42)
        
        # Apply Reweighing preprocessing
        print("Applying Reweighing preprocessing...")
        reweighing = Reweighing(
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )
        
        dataset_train_reweighed = reweighing.fit_transform(dataset_train)
        
        # Train model on reweighed data
        X_train_rw = dataset_train_reweighed.features
        y_train_rw = dataset_train_reweighed.labels.ravel()
        sample_weights = dataset_train_reweighed.instance_weights
        
        rf_model_rw = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_rw.fit(X_train_rw, y_train_rw, sample_weight=sample_weights)
        
        # Make predictions
        X_test = dataset_test.features
        y_pred_rw = rf_model_rw.predict(X_test)
        
        # Create dataset with predictions
        dataset_pred_rw = dataset_test.copy()
        dataset_pred_rw.labels = y_pred_rw
        
        # Calculate metrics for mitigated model
        metric_rw = ClassificationMetric(
            dataset_test,
            dataset_pred_rw,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups
        )
        
        print(f"Reweighed Model - Disparate Impact: {metric_rw.disparate_impact():.4f}")
        print(f"Reweighed Model - Equalized Odds Difference: {metric_rw.equalized_odds_difference():.4f}")
        print(f"Reweighed Model - False Positive Rate Difference: {metric_rw.false_positive_rate_difference():.4f}")
        
        return dataset_test, dataset_pred_rw
    
    def run_complete_audit(self):
        """Run the complete bias audit process"""
        print("COMPAS RECIDIVISM DATASET BIAS AUDIT")
        print("="*60)
        
        # Step 1: Load and prepare data
        df = self.load_and_prepare_data()
        
        # Step 2: Analyze dataset bias
        dataset_metric = self.analyze_dataset_bias()
        
        # Step 3: Train baseline model and analyze prediction bias
        dataset_test, dataset_pred, model = self.train_baseline_model()
        prediction_metric = self.analyze_prediction_bias(dataset_test, dataset_pred)
        
        # Step 4: Generate visualizations
        self.create_visualizations(dataset_test, dataset_pred)
        
        # Step 5: Apply bias mitigation
        dataset_test_mit, dataset_pred_mit = self.apply_bias_mitigation()
        
        print("\n" + "="*60)
        print("AUDIT COMPLETE - CHECK VISUALIZATIONS ABOVE")
        print("="*60)
        
        return {
            'dataset_metric': dataset_metric,
            'prediction_metric': prediction_metric,
            'original_predictions': dataset_pred,
            'mitigated_predictions': dataset_pred_mit,
            'test_data': dataset_test
        }

# Run the complete audit
if __name__ == "__main__":
    auditor = COMPASBiasAuditor()
    results = auditor.run_complete_audit()
    
    print("\nBias audit completed successfully!")
    print("Key findings:")
    print("1. Check the generated visualizations for bias patterns")
    print("2. Review the printed metrics for quantitative bias measures")
    print("3. Compare original vs. mitigated model performance")