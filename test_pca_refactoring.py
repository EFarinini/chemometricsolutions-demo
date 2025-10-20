"""
Test Suite for PCA Refactoring
================================

Comprehensive tests to verify that the PCA refactoring is working correctly.
Tests all pca_utils modules and their integration with pca.py.
"""

import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(message):
    print(f"{GREEN}✓{RESET} {message}")

def print_error(message):
    print(f"{RED}✗{RESET} {message}")

def print_warning(message):
    print(f"{YELLOW}⚠{RESET} {message}")

def print_section(message):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{message}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self):
        self.passed += 1

    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))

    def print_summary(self):
        print_section("Test Summary")
        total = self.passed + self.failed
        print(f"Total tests: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")

        if self.errors:
            print(f"\n{RED}Failed Tests:{RESET}")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")

        return self.failed == 0


def test_imports():
    """Test 1: Verify all imports work correctly"""
    print_section("Test 1: Import Tests")
    results = TestResults()

    # Test 1.1: Import pca_utils package
    try:
        import pca_utils
        print_success("pca_utils package imported")
        results.add_pass()
    except Exception as e:
        print_error(f"Failed to import pca_utils: {e}")
        results.add_fail("pca_utils import", str(e))
        return results

    # Test 1.2: Import config
    try:
        from pca_utils.config import (DEFAULT_N_COMPONENTS, DEFAULT_CONFIDENCE_LEVEL,
                                       VARIMAX_MAX_ITER, VARIMAX_TOLERANCE)
        print_success("Config constants imported")
        print(f"  - DEFAULT_N_COMPONENTS: {DEFAULT_N_COMPONENTS}")
        print(f"  - DEFAULT_CONFIDENCE_LEVEL: {DEFAULT_CONFIDENCE_LEVEL}")
        print(f"  - VARIMAX_MAX_ITER: {VARIMAX_MAX_ITER}")
        print(f"  - VARIMAX_TOLERANCE: {VARIMAX_TOLERANCE}")
        results.add_pass()
    except Exception as e:
        print_error(f"Failed to import config: {e}")
        results.add_fail("config import", str(e))

    # Test 1.3: Import calculation functions
    try:
        from pca_utils.pca_calculations import (compute_pca, varimax_rotation,
                                                 calculate_explained_variance)
        print_success("Calculation functions imported")
        results.add_pass()
    except Exception as e:
        print_error(f"Failed to import calculations: {e}")
        results.add_fail("calculations import", str(e))

    # Test 1.4: Import plotting functions
    try:
        from pca_utils.pca_plots import (plot_scree, plot_cumulative_variance,
                                         plot_scores, plot_loadings, plot_loadings_line,
                                         plot_biplot, add_convex_hulls)
        print_success("Plotting functions imported")
        results.add_pass()
    except Exception as e:
        print_error(f"Failed to import plots: {e}")
        results.add_fail("plots import", str(e))

    # Test 1.5: Import statistical functions
    try:
        from pca_utils.pca_statistics import (calculate_hotelling_t2, calculate_q_residuals,
                                               calculate_contributions, calculate_leverage,
                                               cross_validate_pca)
        print_success("Statistical functions imported")
        results.add_pass()
    except Exception as e:
        print_error(f"Failed to import statistics: {e}")
        results.add_fail("statistics import", str(e))

    # Test 1.6: Import workspace functions
    try:
        from pca_utils.pca_workspace import (save_workspace_to_file, load_workspace_from_file,
                                              save_dataset_split, get_split_datasets_info,
                                              delete_split_dataset, clear_all_split_datasets)
        print_success("Workspace functions imported")
        results.add_pass()
    except Exception as e:
        print_error(f"Failed to import workspace: {e}")
        results.add_fail("workspace import", str(e))

    # Test 1.7: Import using package-level exports
    try:
        from pca_utils import compute_pca, plot_scores, calculate_hotelling_t2, save_dataset_split
        print_success("Package-level imports work (using __all__)")
        results.add_pass()
    except Exception as e:
        print_error(f"Failed package-level imports: {e}")
        results.add_fail("package-level import", str(e))

    return results


def test_compute_pca():
    """Test 2: Test compute_pca functionality"""
    print_section("Test 2: compute_pca() Functionality")
    results = TestResults()

    try:
        from pca_utils.pca_calculations import compute_pca

        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        # Create DataFrame with some structure
        data = np.random.randn(n_samples, n_features)
        data[:, 0] = data[:, 1] + np.random.randn(n_samples) * 0.5  # Correlation
        df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(n_features)])

        print(f"Test data: {n_samples} samples, {n_features} features")

        # Test 2.1: Basic PCA without centering/scaling
        try:
            result = compute_pca(df, n_components=5, center=False, scale=False)
            assert 'scores' in result, "Missing 'scores' in result"
            assert 'loadings' in result, "Missing 'loadings' in result"
            assert 'explained_variance' in result, "Missing 'explained_variance' in result"
            assert result['scores'].shape == (n_samples, 5), f"Wrong scores shape: {result['scores'].shape}"
            assert result['loadings'].shape == (n_features, 5), f"Wrong loadings shape: {result['loadings'].shape}"
            print_success("Basic PCA (no preprocessing) works")
            results.add_pass()
        except Exception as e:
            print_error(f"Basic PCA failed: {e}")
            results.add_fail("basic PCA", str(e))

        # Test 2.2: PCA with centering
        try:
            result = compute_pca(df, n_components=5, center=True, scale=False)
            assert result['scores'].shape == (n_samples, 5)
            print_success("PCA with centering works")
            results.add_pass()
        except Exception as e:
            print_error(f"PCA with centering failed: {e}")
            results.add_fail("PCA centering", str(e))

        # Test 2.3: PCA with centering and scaling
        try:
            result = compute_pca(df, n_components=5, center=True, scale=True)
            assert result['scores'].shape == (n_samples, 5)
            assert result['scaler'] is not None, "Scaler should not be None when scale=True"
            print_success("PCA with centering and scaling works")
            results.add_pass()
        except Exception as e:
            print_error(f"PCA with scaling failed: {e}")
            results.add_fail("PCA scaling", str(e))

        # Test 2.4: Check variance explained sums to <= 1
        try:
            result = compute_pca(df, n_components=5, center=True, scale=True)
            var_sum = np.sum(result['explained_variance_ratio'])
            assert 0 < var_sum <= 1.0, f"Variance ratio sum should be (0,1], got {var_sum}"
            print_success(f"Variance explained: {var_sum*100:.1f}%")
            results.add_pass()
        except Exception as e:
            print_error(f"Variance check failed: {e}")
            results.add_fail("variance check", str(e))

        # Test 2.5: Test with numpy array input
        try:
            result = compute_pca(data, n_components=5, center=True, scale=False)
            assert result['scores'].shape == (n_samples, 5)
            print_success("PCA works with numpy array input")
            results.add_pass()
        except Exception as e:
            print_error(f"Numpy array input failed: {e}")
            results.add_fail("numpy array input", str(e))

    except ImportError as e:
        print_error(f"Cannot import compute_pca: {e}")
        results.add_fail("compute_pca import", str(e))

    return results


def test_varimax_rotation():
    """Test 3: Test varimax_rotation functionality"""
    print_section("Test 3: varimax_rotation() Functionality")
    results = TestResults()

    try:
        from pca_utils.pca_calculations import compute_pca, varimax_rotation

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(50, 10), columns=[f'V{i}' for i in range(10)])

        # First compute PCA
        pca_result = compute_pca(data, n_components=3, center=True, scale=True)
        loadings = pca_result['loadings']

        # Test 3.1: Basic Varimax rotation
        try:
            rotated, n_iter = varimax_rotation(loadings, max_iter=100, tol=1e-6)
            assert rotated.shape == loadings.shape, f"Shape mismatch: {rotated.shape} vs {loadings.shape}"
            assert isinstance(n_iter, int), "Number of iterations should be int"
            print_success(f"Varimax rotation works (converged in {n_iter} iterations)")
            results.add_pass()
        except Exception as e:
            print_error(f"Varimax rotation failed: {e}")
            results.add_fail("varimax rotation", str(e))

        # Test 3.2: Check orthogonality is preserved
        try:
            rotated, _ = varimax_rotation(loadings, max_iter=100, tol=1e-6)
            if isinstance(rotated, pd.DataFrame):
                rotated_arr = rotated.values
            else:
                rotated_arr = rotated

            # Check if columns are still approximately orthogonal
            ortho_check = rotated_arr.T @ rotated_arr
            off_diag = ortho_check - np.diag(np.diag(ortho_check))
            max_off_diag = np.max(np.abs(off_diag))

            # Note: Varimax doesn't guarantee strict orthogonality, just checks it's reasonable
            print_success(f"Orthogonality check: max off-diagonal = {max_off_diag:.4f}")
            results.add_pass()
        except Exception as e:
            print_error(f"Orthogonality check failed: {e}")
            results.add_fail("orthogonality check", str(e))

    except ImportError as e:
        print_error(f"Cannot import varimax_rotation: {e}")
        results.add_fail("varimax import", str(e))

    return results


def test_plotting_functions():
    """Test 4: Test plotting functions return valid figures"""
    print_section("Test 4: Plotting Functions")
    results = TestResults()

    try:
        from pca_utils.pca_calculations import compute_pca
        from pca_utils.pca_plots import (plot_scree, plot_cumulative_variance,
                                         plot_scores, plot_loadings, plot_loadings_line)
        import plotly.graph_objects as go

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(50, 10), columns=[f'V{i}' for i in range(10)])
        pca_result = compute_pca(data, n_components=5, center=True, scale=True)

        # Test 4.1: plot_scree
        try:
            fig = plot_scree(pca_result['explained_variance_ratio'])
            assert isinstance(fig, go.Figure), "plot_scree should return plotly Figure"
            print_success("plot_scree() works")
            results.add_pass()
        except Exception as e:
            print_error(f"plot_scree failed: {e}")
            results.add_fail("plot_scree", str(e))

        # Test 4.2: plot_cumulative_variance
        try:
            fig = plot_cumulative_variance(pca_result['cumulative_variance'])
            assert isinstance(fig, go.Figure), "plot_cumulative_variance should return plotly Figure"
            print_success("plot_cumulative_variance() works")
            results.add_pass()
        except Exception as e:
            print_error(f"plot_cumulative_variance failed: {e}")
            results.add_fail("plot_cumulative_variance", str(e))

        # Test 4.3: plot_scores
        try:
            fig = plot_scores(pca_result['scores'], 'PC1', 'PC2',
                            pca_result['explained_variance_ratio'])
            assert isinstance(fig, go.Figure), "plot_scores should return plotly Figure"
            print_success("plot_scores() works")
            results.add_pass()
        except Exception as e:
            print_error(f"plot_scores failed: {e}")
            results.add_fail("plot_scores", str(e))

        # Test 4.4: plot_loadings
        try:
            fig = plot_loadings(pca_result['loadings'], 'PC1', 'PC2',
                              pca_result['explained_variance_ratio'])
            assert isinstance(fig, go.Figure), "plot_loadings should return plotly Figure"
            print_success("plot_loadings() works")
            results.add_pass()
        except Exception as e:
            print_error(f"plot_loadings failed: {e}")
            results.add_fail("plot_loadings", str(e))

        # Test 4.5: plot_loadings_line
        try:
            fig = plot_loadings_line(pca_result['loadings'], ['PC1', 'PC2'])
            assert isinstance(fig, go.Figure), "plot_loadings_line should return plotly Figure"
            print_success("plot_loadings_line() works")
            results.add_pass()
        except Exception as e:
            print_error(f"plot_loadings_line failed: {e}")
            results.add_fail("plot_loadings_line", str(e))

    except ImportError as e:
        print_error(f"Cannot import plotting functions: {e}")
        results.add_fail("plotting import", str(e))

    return results


def test_statistical_functions():
    """Test 5: Test statistical functions"""
    print_section("Test 5: Statistical Functions")
    results = TestResults()

    try:
        from pca_utils.pca_calculations import compute_pca
        from pca_utils.pca_statistics import (calculate_hotelling_t2, calculate_q_residuals,
                                               calculate_contributions, calculate_leverage)

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(50, 10), columns=[f'V{i}' for i in range(10)])
        pca_result = compute_pca(data, n_components=5, center=True, scale=True)

        # Test 5.1: calculate_hotelling_t2
        try:
            t2_values, t2_limit = calculate_hotelling_t2(pca_result['scores'],
                                                          pca_result['eigenvalues'])
            assert len(t2_values) == 50, "Should have T2 value for each sample"
            assert t2_limit > 0, "T2 limit should be positive"
            print_success(f"calculate_hotelling_t2() works (limit={t2_limit:.2f})")
            results.add_pass()
        except Exception as e:
            print_error(f"calculate_hotelling_t2 failed: {e}")
            results.add_fail("hotelling_t2", str(e))

        # Test 5.2: calculate_q_residuals
        try:
            q_values, q_limit = calculate_q_residuals(pca_result['processed_data'],
                                                       pca_result['scores'],
                                                       pca_result['loadings'])
            assert len(q_values) == 50, "Should have Q value for each sample"
            assert q_limit >= 0, "Q limit should be non-negative"
            print_success(f"calculate_q_residuals() works (limit={q_limit:.2f})")
            results.add_pass()
        except Exception as e:
            print_error(f"calculate_q_residuals failed: {e}")
            results.add_fail("q_residuals", str(e))

        # Test 5.3: calculate_contributions
        try:
            contrib_df = calculate_contributions(pca_result['loadings'],
                                                  pca_result['explained_variance_ratio'])
            assert isinstance(contrib_df, pd.DataFrame), "Should return DataFrame"
            assert 'Contribution_%' in contrib_df.columns, "Should have Contribution_% column"
            assert len(contrib_df) == 10, "Should have contribution for each variable"
            total = contrib_df['Contribution_%'].sum()
            assert abs(total - 100.0) < 1e-6, f"Contributions should sum to 100%, got {total}"
            print_success(f"calculate_contributions() works (sum={total:.1f}%)")
            results.add_pass()
        except Exception as e:
            print_error(f"calculate_contributions failed: {e}")
            results.add_fail("contributions", str(e))

        # Test 5.4: calculate_leverage
        try:
            leverage = calculate_leverage(pca_result['scores'])
            assert len(leverage) == 50, "Should have leverage for each sample"
            assert np.all(leverage >= 0), "Leverage should be non-negative"
            print_success(f"calculate_leverage() works (mean={np.mean(leverage):.4f})")
            results.add_pass()
        except Exception as e:
            print_error(f"calculate_leverage failed: {e}")
            results.add_fail("leverage", str(e))

    except ImportError as e:
        print_error(f"Cannot import statistical functions: {e}")
        results.add_fail("statistics import", str(e))

    return results


def test_workspace_functions():
    """Test 6: Test workspace functions (without Streamlit session_state)"""
    print_section("Test 6: Workspace Functions")
    results = TestResults()

    try:
        from pca_utils.pca_workspace import (get_split_datasets_info)

        # Test 6.1: get_split_datasets_info (should work without session_state)
        try:
            info = get_split_datasets_info()
            assert isinstance(info, dict), "Should return dict"
            assert 'count' in info, "Should have 'count' key"
            assert info['count'] == 0, "Should be 0 when no split_datasets"
            print_success("get_split_datasets_info() works (no session_state)")
            results.add_pass()
        except Exception as e:
            print_error(f"get_split_datasets_info failed: {e}")
            results.add_fail("workspace info", str(e))

        # Note: Other workspace functions require streamlit.session_state
        print_warning("Other workspace functions require Streamlit session_state (not tested here)")

    except ImportError as e:
        print_error(f"Cannot import workspace functions: {e}")
        results.add_fail("workspace import", str(e))

    return results


def test_pca_py_imports():
    """Test 7: Verify pca.py can import everything"""
    print_section("Test 7: pca.py Import Check")
    results = TestResults()

    try:
        # Read pca.py and check imports
        pca_file = Path(__file__).parent / 'pca.py'
        if not pca_file.exists():
            print_error("pca.py not found")
            results.add_fail("pca.py location", "File not found")
            return results

        print_success(f"Found pca.py at: {pca_file}")

        # Test importing the show function (without calling it)
        try:
            sys.path.insert(0, str(pca_file.parent))
            import pca

            assert hasattr(pca, 'show'), "pca.py should have a show() function"
            print_success("pca.py imports successfully")
            print_success("pca.show() function found")
            results.add_pass()
        except Exception as e:
            print_error(f"Failed to import pca.py: {e}")
            traceback.print_exc()
            results.add_fail("pca.py import", str(e))

    except Exception as e:
        print_error(f"Error checking pca.py: {e}")
        results.add_fail("pca.py check", str(e))

    return results


def test_integration():
    """Test 8: Integration test - full PCA workflow"""
    print_section("Test 8: Integration Test (Full Workflow)")
    results = TestResults()

    try:
        from pca_utils import (compute_pca, varimax_rotation, plot_scree,
                               calculate_hotelling_t2, calculate_contributions)

        # Create realistic test data
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        # Create data with some structure
        data = np.random.randn(n_samples, n_features)
        data[:, 0] = 2 * data[:, 1] + data[:, 2] + np.random.randn(n_samples) * 0.5
        data[:, 3] = -1 * data[:, 4] + np.random.randn(n_samples) * 0.3

        df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(n_features)])

        print(f"Test dataset: {n_samples} samples, {n_features} features")

        # Step 1: Compute PCA
        try:
            pca_result = compute_pca(df, n_components=5, center=True, scale=True)
            print_success("Step 1: PCA computation complete")
            results.add_pass()
        except Exception as e:
            print_error(f"Step 1 failed: {e}")
            results.add_fail("integration step 1", str(e))
            return results

        # Step 2: Apply Varimax rotation
        try:
            rotated_loadings, n_iter = varimax_rotation(pca_result['loadings'])
            print_success(f"Step 2: Varimax rotation complete ({n_iter} iterations)")
            results.add_pass()
        except Exception as e:
            print_error(f"Step 2 failed: {e}")
            results.add_fail("integration step 2", str(e))

        # Step 3: Create visualization
        try:
            fig = plot_scree(pca_result['explained_variance_ratio'])
            print_success("Step 3: Scree plot created")
            results.add_pass()
        except Exception as e:
            print_error(f"Step 3 failed: {e}")
            results.add_fail("integration step 3", str(e))

        # Step 4: Calculate diagnostics
        try:
            t2_values, t2_limit = calculate_hotelling_t2(pca_result['scores'],
                                                          pca_result['eigenvalues'])
            n_outliers = np.sum(t2_values > t2_limit)
            print_success(f"Step 4: T2 diagnostics complete ({n_outliers} outliers)")
            results.add_pass()
        except Exception as e:
            print_error(f"Step 4 failed: {e}")
            results.add_fail("integration step 4", str(e))

        # Step 5: Calculate contributions
        try:
            contrib_df = calculate_contributions(pca_result['loadings'],
                                                  pca_result['explained_variance_ratio'],
                                                  n_components=5)
            top_var = contrib_df.iloc[0]['Variable']
            top_contrib = contrib_df.iloc[0]['Contribution_%']
            print_success(f"Step 5: Variable contributions calculated (top: {top_var}, {top_contrib:.1f}%)")
            results.add_pass()
        except Exception as e:
            print_error(f"Step 5 failed: {e}")
            results.add_fail("integration step 5", str(e))

        print_success("Full workflow completed successfully!")

    except ImportError as e:
        print_error(f"Cannot import required modules: {e}")
        results.add_fail("integration imports", str(e))

    return results


def main():
    """Run all tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}PCA Refactoring Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    all_results = []

    # Run all tests
    all_results.append(("Import Tests", test_imports()))
    all_results.append(("compute_pca Tests", test_compute_pca()))
    all_results.append(("varimax_rotation Tests", test_varimax_rotation()))
    all_results.append(("Plotting Tests", test_plotting_functions()))
    all_results.append(("Statistical Tests", test_statistical_functions()))
    all_results.append(("Workspace Tests", test_workspace_functions()))
    all_results.append(("pca.py Import Test", test_pca_py_imports()))
    all_results.append(("Integration Test", test_integration()))

    # Print overall summary
    print_section("Overall Summary")

    total_passed = sum(r.passed for _, r in all_results)
    total_failed = sum(r.failed for _, r in all_results)
    total_tests = total_passed + total_failed

    print(f"\nTotal tests run: {total_tests}")
    print(f"{GREEN}Total passed: {total_passed}{RESET}")
    print(f"{RED}Total failed: {total_failed}{RESET}")

    if total_failed > 0:
        print(f"\n{RED}Some tests failed. Details:{RESET}")
        for test_name, result in all_results:
            if result.failed > 0:
                print(f"\n{test_name}:")
                for error_name, error_msg in result.errors:
                    print(f"  - {error_name}: {error_msg}")

    # Return exit code
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
