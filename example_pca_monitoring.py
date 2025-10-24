"""
Example: PCA Process Monitoring with Fault Detection

This script demonstrates how to use the PCAMonitor class for:
1. Training a PCA monitoring model from normal operating data
2. Calculating T² and Q statistics with multiple control limits
3. Detecting faults in new process data
4. Analyzing fault contributions for diagnosis
5. Saving and loading trained models

Author: ChemometricSolutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pca_utils import PCAMonitor

# Set random seed for reproducibility
np.random.seed(42)

def generate_example_data():
    """
    Generate synthetic process data for demonstration.

    Returns
    -------
    X_train : pd.DataFrame
        Normal operating condition data (200 samples, 10 variables)
    X_test : pd.DataFrame
        Test data with some faulty samples (100 samples, 10 variables)
    """
    # Training data: Normal Operating Conditions (NOC)
    n_train = 200
    n_vars = 10

    # Generate correlated normal data
    mean = np.zeros(n_vars)
    cov = np.eye(n_vars)
    # Add some correlations
    for i in range(n_vars - 1):
        cov[i, i+1] = 0.7
        cov[i+1, i] = 0.7

    X_train = np.random.multivariate_normal(mean, cov, n_train)

    # Test data: Mix of normal and faulty samples
    n_test = 100
    X_test = np.random.multivariate_normal(mean, cov, n_test)

    # Inject faults into test data
    # Fault 1: Mean shift (samples 20-29)
    X_test[20:30, 0:3] += 3.0  # Shift in first 3 variables

    # Fault 2: Variance increase (samples 50-59)
    X_test[50:60, 4:7] *= 2.5  # Increased variance in vars 4-6

    # Fault 3: Correlation break (samples 80-89)
    X_test[80:90, 8] = np.random.randn(10) * 5  # Break correlation in var 8

    # Convert to DataFrames with variable names
    var_names = [f'Temperature_{i+1}' if i < 3 else f'Pressure_{i-2}' if i < 6 else f'Flow_{i-5}'
                 for i in range(n_vars)]

    X_train_df = pd.DataFrame(X_train, columns=var_names)
    X_test_df = pd.DataFrame(X_test, columns=var_names)

    return X_train_df, X_test_df


def example_basic_monitoring():
    """Example 1: Basic PCA monitoring workflow"""
    print("=" * 70)
    print("Example 1: Basic PCA Monitoring")
    print("=" * 70)

    # Generate data
    X_train, X_test = generate_example_data()
    print(f"\nTraining data: {X_train.shape[0]} samples, {X_train.shape[1]} variables")
    print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} variables")

    # Create and train PCA monitor
    print("\n1. Training PCA monitoring model...")
    monitor = PCAMonitor(
        n_components=5,  # Use 5 principal components
        scaling='auto',  # Standardization
        alpha_levels=[0.975, 0.995, 0.9995]  # 97.5%, 99.5%, 99.95% control limits
    )

    monitor.fit(X_train)

    # Display model summary
    summary = monitor.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  - Components: {summary['n_components']}")
    print(f"  - Variance explained: {summary['variance_explained']*100:.2f}%")
    print(f"  - Training samples: {summary['n_samples_train']}")
    print(f"\nControl Limits:")
    print(f"  T² limits: {summary['t2_limits']}")
    print(f"  Q limits: {summary['q_limits']}")

    # Test on new data
    print("\n2. Testing new data...")
    results = monitor.predict(X_test, return_contributions=True)

    # Analyze results
    n_faults = results['faults'].sum()
    print(f"\nFault Detection Results:")
    print(f"  - Total samples tested: {len(results['faults'])}")
    print(f"  - Faults detected: {n_faults} ({n_faults/len(results['faults'])*100:.1f}%)")

    # Count fault types
    fault_types = pd.Series(results['fault_type']).value_counts()
    print(f"\nFault Type Distribution:")
    for fault_type, count in fault_types.items():
        print(f"  - {fault_type}: {count}")

    # Generate summary table
    summary_df = monitor.get_fault_summary(results)
    print(f"\nFirst 10 samples summary:")
    print(summary_df.head(10))

    # Save monitoring chart
    print("\n3. Creating monitoring chart...")
    fig = monitor.plot_monitoring_chart(results, title="PCA Process Monitoring - Example 1")
    fig.write_html("monitoring_chart.html")
    print("   Saved to: monitoring_chart.html")

    return monitor, results


def example_fault_diagnosis(monitor, results):
    """Example 2: Fault diagnosis using contribution plots"""
    print("\n" + "=" * 70)
    print("Example 2: Fault Diagnosis with Contribution Analysis")
    print("=" * 70)

    # Find faulty samples
    faulty_indices = np.where(results['faults'])[0]
    print(f"\nFound {len(faulty_indices)} faulty samples")

    # Analyze first few faults
    print("\nAnalyzing first 3 detected faults:")
    for i, idx in enumerate(faulty_indices[:3]):
        print(f"\n--- Fault {i+1}: Sample {idx+1} ---")
        print(f"T² = {results['t2'][idx]:.2f} (limit: {results['t2_limits'][0.975]:.2f})")
        print(f"Q = {results['q'][idx]:.2f} (limit: {results['q_limits'][0.975]:.2f})")
        print(f"Type: {results['fault_type'][idx]}")

        # Get top contributing variables for Q statistic
        q_contrib = results['contributions_q'][idx]
        top_vars_idx = np.argsort(q_contrib)[-3:][::-1]

        print(f"\nTop 3 contributing variables (Q):")
        for rank, var_idx in enumerate(top_vars_idx, 1):
            var_name = monitor.feature_names_[var_idx]
            contrib = q_contrib[var_idx]
            print(f"  {rank}. {var_name}: {contrib:.3f}")

        # Create contribution plot
        fig = monitor.plot_contribution_chart(
            results,
            sample_idx=idx,
            statistic='q',
            top_n=10
        )
        filename = f"contribution_sample_{idx+1}.html"
        fig.write_html(filename)
        print(f"   Contribution plot saved to: {filename}")


def example_save_load_model():
    """Example 3: Save and load trained models"""
    print("\n" + "=" * 70)
    print("Example 3: Save and Load Monitoring Model")
    print("=" * 70)

    # Generate data and train model
    X_train, X_test = generate_example_data()

    print("\n1. Training new model...")
    monitor = PCAMonitor(n_components=5, scaling='auto')
    monitor.fit(X_train)

    # Save model
    model_path = "pca_monitor_model.pkl"
    print(f"\n2. Saving model to {model_path}...")
    monitor.save(model_path)

    # Load model
    print(f"\n3. Loading model from {model_path}...")
    monitor_loaded = PCAMonitor.load(model_path)

    # Test loaded model
    print("\n4. Testing loaded model...")
    results = monitor_loaded.predict(X_test)
    print(f"   Detected {results['faults'].sum()} faults")

    print("\n   Model successfully saved and loaded!")

    return monitor_loaded


def example_combined_chart(monitor, results):
    """Example 4: Combined T² vs Q scatter plot"""
    print("\n" + "=" * 70)
    print("Example 4: Combined T² vs Q Monitoring Chart")
    print("=" * 70)

    from pca_utils import plot_combined_monitoring_chart

    # Create sample labels
    sample_labels = [f"Sample {i+1}" for i in range(len(results['t2']))]

    # Create combined chart
    print("\nCreating combined T² vs Q scatter plot...")
    fig = plot_combined_monitoring_chart(
        results,
        results['t2_limits'],
        results['q_limits'],
        sample_labels=sample_labels,
        title="T² vs Q Chart - Fault Detection Regions"
    )

    fig.write_html("t2_vs_q_chart.html")
    print("   Saved to: t2_vs_q_chart.html")


def example_different_confidence_levels():
    """Example 5: Using different confidence levels"""
    print("\n" + "=" * 70)
    print("Example 5: Multiple Confidence Levels")
    print("=" * 70)

    X_train, X_test = generate_example_data()

    # Train with multiple confidence levels
    monitor = PCAMonitor(
        n_components=5,
        scaling='auto',
        alpha_levels=[0.95, 0.975, 0.99, 0.995, 0.9995]
    )

    monitor.fit(X_train)
    results = monitor.predict(X_test)

    print("\nFault detection at different confidence levels:")
    for alpha in monitor.alpha_levels:
        t2_exceeds = (results['t2'] > results['t2_limits'][alpha]).sum()
        q_exceeds = (results['q'] > results['q_limits'][alpha]).sum()
        total_faults = ((results['t2'] > results['t2_limits'][alpha]) |
                       (results['q'] > results['q_limits'][alpha])).sum()

        print(f"\n{alpha*100:.2f}% Confidence Level:")
        print(f"  - T² limit: {results['t2_limits'][alpha]:.2f} ({t2_exceeds} exceedances)")
        print(f"  - Q limit: {results['q_limits'][alpha]:.2f} ({q_exceeds} exceedances)")
        print(f"  - Total faults: {total_faults}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("PCA PROCESS MONITORING - COMPREHENSIVE EXAMPLES")
    print("=" * 70)

    # Example 1: Basic monitoring
    monitor, results = example_basic_monitoring()

    # Example 2: Fault diagnosis
    example_fault_diagnosis(monitor, results)

    # Example 3: Save/load
    monitor_loaded = example_save_load_model()

    # Example 4: Combined chart
    example_combined_chart(monitor, results)

    # Example 5: Different confidence levels
    example_different_confidence_levels()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - monitoring_chart.html")
    print("  - contribution_sample_*.html")
    print("  - t2_vs_q_chart.html")
    print("  - pca_monitor_model.pkl")
    print("\nOpen the HTML files in your browser to view interactive charts.")


if __name__ == "__main__":
    main()
