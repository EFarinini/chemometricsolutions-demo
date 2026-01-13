"""
🧬 Genetic Algorithm Variable Selection - Leardi GAPLSOPT + GAPLSSP

Minimal Streamlit interface for:
- Step 1: Randomization test to find optimal evaluations (GAPLSOPT)
- Step 2: Production GA with probability evolution (GAPLSSP)
- Graphics: Spectral bands, selection frequency, CV curves

Based on:
- Leardi R. et al. (2002) "Variable selection for multivariate calibration
  using a genetic algorithm" - Analytica Chimica Acta, 461, 189-200
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS - GA MODULE
# ═══════════════════════════════════════════════════════════════════════════

try:
    from ga_variable_selection import gaplsopt, GAPLSSP
    from ga_variable_selection import (
        plot_spectrum_with_bands,
        plot_selection_frequency,
        plot_fitness_evolution,
        plot_cv_curve,
        plot_rmsecv_curve,
        create_bands_table
    )
    from ga_variable_selection.results_visualization import (
        plot_spectra_with_consensus_bands,
        plot_multirun_band_selection,
        plot_selection_frequency_spectral_order,
        export_selected_dataset
    )
    GA_AVAILABLE = True
except ImportError as e:
    GA_AVAILABLE = False
    import_error = str(e)

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS - WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════

try:
    from workspace_utils import get_workspace_datasets, display_workspace_dataset_selector
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# NOTE: st.set_page_config() is called in streamlit_app.py, not here!

# ═══════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def pls_fitness(X_subset: np.ndarray, y_vals: np.ndarray) -> float:
    """
    PLS fitness function for GAPLSOPT/GAPLSSP (Leardi format).

    Returns % variance explained (R² × 100).

    Parameters
    ----------
    X_subset : np.ndarray
        Subset of features (selected variables only)
    y_vals : np.ndarray
        Target values

    Returns
    -------
    float
        R² cross-validation score as percentage (0-100)
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score

    try:
        # Determine number of components (min of 5, n_features, n_samples-1)
        n_components = min(5, X_subset.shape[1], X_subset.shape[0] - 1)

        if n_components < 1:
            return 0.0

        # PLS model with cross-validation
        model = PLSRegression(n_components=n_components)
        r2 = cross_val_score(model, X_subset, y_vals, cv=5, scoring='r2').mean()

        # Convert to % and ensure non-negative
        return max(0, r2 * 100)

    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def show():
    """Main page function - called by homepage routing."""

    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION STATE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    if 'gaplsopt_results' not in st.session_state:
        st.session_state['gaplsopt_results'] = None
    if 'gaplssp_results' not in st.session_state:
        st.session_state['gaplssp_results'] = None
    if 'recommended_evals' not in st.session_state:
        st.session_state['recommended_evals'] = 200

    # ═══════════════════════════════════════════════════════════════════════════
    # DEPENDENCY CHECK
    # ═══════════════════════════════════════════════════════════════════════════

    if not GA_AVAILABLE:
        st.error(f"❌ GA module not available: {import_error}")
        st.info("Install: `pip install -e .` from project root")
        st.stop()

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE HEADER
    # ═══════════════════════════════════════════════════════════════════════════

    st.title("🧬 Genetic Algorithm Variable Selection")

    st.markdown("""
    **Leardi GAPLSOPT + GAPLSSP** - Faithful Python implementation

    - **Step 1**: Dataset loaded ✅
    - **Step 2**: Select variables & samples (user-controlled)
    - **Step 3**: Select target variable (user-controlled)
    - **Step 4**: Randomization test (optional)
    - **Step 5**: Production GA (required)
    """)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════

    if WORKSPACE_AVAILABLE:
        dataset_result = display_workspace_dataset_selector(
            label="Select dataset for GA variable selection:",
            key="ga_dataset_selector",
            help_text="Choose a dataset from your workspace",
            show_info=True
        )

        if dataset_result is None:
            st.info("💡 **No datasets available**\n\nLoad data in the **Data Handling** page first, then return here.")
            if st.button("→ Go to Data Handling"):
                st.session_state.current_page = "Data Handling"
                st.rerun()
            st.stop()

        dataset_name, df = dataset_result
        st.success(f"✓ Dataset loaded: **{dataset_name}** ({df.shape[0]} samples × {df.shape[1]} variables)")

    else:
        st.warning("⚠️ Workspace module not found - using demo data")
        np.random.seed(42)
        demo_df = pd.DataFrame(
            np.random.randn(100, 51),
            columns=[f"Var_{i}" for i in range(50)] + ["Target"]
        )
        df = demo_df
        dataset_name = "Demo Data"



    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: VARIABLE & SAMPLE SELECTION
    # ═══════════════════════════════════════════════════════════════════════════

    st.header("🎯 Step 2: Variable & Sample Selection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Variables (Columns)")
        col_range = st.columns(2)
        with col_range[0]:
            first_col = st.number_input("First column:", 1, df.shape[1], 1, key="ga_fc")
        with col_range[1]:
            last_col = st.number_input("Last column:", 1, df.shape[1], df.shape[1], key="ga_lc")

        if first_col > last_col:
            st.error("❌ First must be ≤ Last")
            st.stop()

        st.info(f"✓ {last_col - first_col + 1} variables selected")

    with col2:
        st.subheader("👥 Samples (Rows)")
        row_range = st.columns(2)
        with row_range[0]:
            first_row = st.number_input("First sample:", 1, df.shape[0], 1, key="ga_fr")
        with row_range[1]:
            last_row = st.number_input("Last sample:", 1, df.shape[0], df.shape[0], key="ga_lr")

        if first_row > last_row:
            st.error("❌ First must be ≤ Last")
            st.stop()

        st.info(f"✓ {last_row - first_row + 1} samples selected")

    # Select data
    df_selected = df.iloc[first_row-1:last_row, first_col-1:last_col]

    with st.expander("👁️ Preview", expanded=False):
        st.dataframe(df_selected.head(10), use_container_width=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: TARGET VARIABLE SELECTION
    # ═══════════════════════════════════════════════════════════════════════════

    st.header("📌 Step 3: Target Variable Selection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Method")
        st.info("**PLS Regression** (Leardi standard)")
        st.caption("Fitness = R² × 100 (5-fold CV)")

    with col2:
        st.subheader("📊 Target Column")
        all_cols = list(df_selected.columns)
        target_var = st.selectbox("Select target (y):", all_cols, key="ga_target")

        if target_var:
            y = df_selected[target_var].values

            # Validate
            if np.var(y) == 0:
                st.error("❌ Zero variance in target")
                st.stop()

            st.success(f"✓ Target: μ={np.mean(y):.2f}, σ={np.std(y):.2f}")

    # Extract X (all columns except target)
    X = df_selected.drop(columns=[target_var]).values

    # Store original column names for export
    st.session_state['y_column_name'] = target_var
    st.session_state['feature_names'] = [str(col) for col in df_selected.drop(columns=[target_var]).columns.tolist()]

    # Display metrics
    st.markdown("---")
    with st.expander("📊 Data Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features (X)", X.shape[1])
        with col2:
            st.metric("Samples", X.shape[0])
        with col3:
            st.metric("Target range", f"{y.min():.2f} to {y.max():.2f}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: GAPLSOPT (Randomization Test)
    # ═══════════════════════════════════════════════════════════════════════════

    st.header("Step 4: Randomization Test (GAPLSOPT)")

    st.markdown("""
    Determines the optimal number of evaluations by comparing:
    - **True runs**: GA applied to actual data (50 runs)
    - **Random runs**: GA applied to shuffled/random target (50 runs)
    - **Difference**: Identifies when GA achieves genuine variable selection

    This prevents overfitting and finds the "plateau" point where
    true data fitness diverges from random baseline.

    **Fixed parameters (Leardi standard):**
    - Total runs: **100** (50 TRUE + 50 SHUFFLED)
    - Max evaluations: **200** per run
    - These values are FIXED in the original MATLAB code
    """)

    # Run button
    run_gaplsopt = st.button(
        "▶️ Run Randomization Test (100 runs × 200 evals)",
        key="run_gaplsopt_button",
        use_container_width=True
    )

    # Execute GAPLSOPT
    if run_gaplsopt:
        progress_placeholder = st.empty()
        progress_placeholder.info("⏳ Running 100 runs (50 TRUE + 50 SHUFFLED)... this may take 2-3 minutes")

        try:
            results_opt = gaplsopt(
                X, y,
                pls_fitness,  # Fitness function
                test_type='optimization',
                population_size=30,
                cv_groups=5,
                plot=False
            )

            progress_placeholder.success("✅ Randomization test complete!")

            # Store results
            st.session_state['gaplsopt_results'] = results_opt
            st.session_state['recommended_evals'] = results_opt['recommended_evals']

        except Exception as e:
            progress_placeholder.error(f"❌ Error: {str(e)}")
            st.stop()

    # Display results
    if st.session_state['gaplsopt_results'] is not None:
        results_opt = st.session_state['gaplsopt_results']

        st.subheader("📈 Results")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # mean_true is array, take mean of all values
            mean_true_val = np.mean(results_opt['mean_true'])
            st.metric("True Mean", f"{mean_true_val:.2f}%")
        with col2:
            # mean_random is array, take mean of all values
            mean_random_val = np.mean(results_opt['mean_random'])
            st.metric("Random Mean", f"{mean_random_val:.2f}%")
        with col3:
            # max_difference is scalar
            st.metric("Max Difference", f"{results_opt['max_difference']:.2f}%")
        with col4:
            st.metric(
                "✅ Recommended Evals",
                results_opt['recommended_evals'],
                delta_color="inverse"
            )

        # Difference curve
        st.subheader("Difference Curve (True - Random)")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(results_opt['difference']))),
            y=results_opt['difference'],
            mode='lines',
            fill='tozeroy',
            name='Difference',
            line=dict(color='#1f77b4', width=2),
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        # Add vertical line at recommended point
        max_idx = np.argmax(results_opt['difference'])
        fig.add_vline(
            x=max_idx,
            line_dash='dash',
            line_color='red',
            annotation_text=f"Max at {max_idx}",
            annotation_position="top"
        )

        fig.update_layout(
            title="Difference between True and Random Runs",
            xaxis_title="Evaluation Number",
            yaxis_title="Fitness Difference (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Use the recommended evaluations value for Step 5 (Production GA)")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: GAPLSSP (Production GA)
    # ═══════════════════════════════════════════════════════════════════════════

    st.header("Step 5: Production GA (GAPLSSP)")

    st.markdown("""
    Full variable selection with probability evolution:
    - **100 sequential runs**: Probabilities adapt based on selection frequency
    - **Backward elimination**: Every 100 evaluations to refine selection
    - **Stepwise refinement**: Final F-test criterion for model selection
    - **Output**: Selected variables, frequency histogram, CV curve
    """)

    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_runs_ga = st.slider(
                "Number of runs",
                min_value=50,
                max_value=200,
                value=100,
                step=10,
                help="Sequential GA runs (Leardi standard: 100)"
            )

        with col2:
            default_evals = st.session_state.get('recommended_evals', 200)
            n_evals_ga = st.slider(
                "Evaluations per run",
                min_value=50,
                max_value=500,
                value=default_evals,
                step=10,
                help="From Step 1 recommendation"
            )

        with col3:
            population_size = st.slider(
                "Population size",
                min_value=10,
                max_value=50,
                value=30,
                step=5,
                help="Leardi standard: 30"
            )

        with col4:
            cv_groups = st.slider(
                "CV groups",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                help="Cross-validation groups (Leardi: 5)"
            )

    # Number of independent runs configuration
    st.markdown("---")
    st.subheader("🔄 Independent Runs & Consensus")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Multiple independent GAPLSSP runs** ensure robust variable selection.
        Each run uses a different random seed to explore different solutions.
        """)

    with col2:
        n_independent_runs = st.slider(
            label="Number of Independent Runs",
            min_value=1,
            max_value=5,
            value=5,
            step=1,
            help=(
                "**1 run**: Single execution (no consensus)\n\n"
                "**3 runs**: Minimum for consensus (≥2/3 = 67%)\n\n"
                "**5 runs**: Standard consensus (≥3/5 = 60%) ← RECOMMENDED"
            ),
            key="n_independent_runs_slider"
        )

    # Calculate consensus threshold based on number of runs
    if n_independent_runs == 1:
        consensus_threshold = 1
        threshold_text = "≥1/1 (single run, no consensus)"
    elif n_independent_runs == 2:
        consensus_threshold = 2
        threshold_text = "≥2/2 (both must agree = 100%)"
    elif n_independent_runs == 3:
        consensus_threshold = 2
        threshold_text = "≥2/3 runs (67% agreement)"
    elif n_independent_runs == 4:
        consensus_threshold = 3
        threshold_text = "≥3/4 runs (75% agreement)"
    else:  # 5 runs
        consensus_threshold = 3
        threshold_text = "≥3/5 runs (60% agreement)"

    # Display info about selected configuration
    st.info(
        f"🔄 Will execute **{n_independent_runs} independent GAPLSSP run{'s' if n_independent_runs != 1 else ''}** "
        f"with different random seeds. "
        f"Consensus threshold: **{threshold_text}**"
    )

    # Run button
    run_gaplssp = st.button(
        "▶️ Run Variable Selection",
        key="run_gaplssp_button",
        use_container_width=True
    )

    # Execute GAPLSSP - N Independent Runs with Consensus
    if run_gaplssp:
        st.subheader(f"🧬 Running {n_independent_runs} Independent GAPLSSP Execution{'s' if n_independent_runs != 1 else ''}")

        # Storage for all runs
        all_runs_results = []
        all_selected_variables = []

        # Progress tracking
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_log_placeholder = st.empty()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Execute N independent runs
        # ═══════════════════════════════════════════════════════════════

        try:
            for run_num in range(1, n_independent_runs + 1):
                progress_text.text(f"⏳ Executing independent run {run_num}/{n_independent_runs}...")
                progress_bar.progress(run_num / (n_independent_runs + 1))

                # Initialize progress log for this run
                progress_log = []
                update_counter = [0]

                def progress_callback(event_type, data):
                    """Callback for real-time progress updates (Leardi style)."""
                    if event_type == 'run_start':
                        run = data['run']
                        progress_log.append(f"\n{'='*70}")
                        progress_log.append(f"Run {run}")
                        progress_log.append(f"{data['fitness_min']:.7f} - {data['fitness_max']:.7f}")

                    elif event_type == 'initial_pop':
                        fitness = data['fitness']
                        progress_log.append(f"\nAfter creation of original population: {fitness:.4f}")

                    elif event_type == 'evaluation':
                        ev_num = data['evaluation']
                        fitness = data['fitness']
                        progress_log.append(f"ev. {ev_num:3d} - {fitness:.4f}")

                    # Update display every 10 events for performance
                    update_counter[0] += 1
                    if update_counter[0] % 10 == 0:
                        display_text = "\n".join(progress_log[-30:])  # Last 30 lines
                        progress_log_placeholder.code(display_text, language=None)

                # CRITICAL: Different random seed for each run
                # This ensures true independence
                ga = GAPLSSP(
                    X, y,
                    pls_fitness,
                    n_runs=n_runs_ga,
                    n_evals=n_evals_ga,
                    population_size=population_size,
                    cv_groups=cv_groups,
                    nvar_avg=5,
                    max_vars=30,
                    verbose=False,
                    progress_callback=progress_callback,
                    random_seed=42 + run_num  # ← DIFFERENT SEED EACH RUN
                )

                # Execute
                results = ga.run()

                # Store
                all_runs_results.append(results)
                all_selected_variables.append(
                    set(results['selected_variables'])
                )

                # Update progress
                progress_log.append(f"\n{'='*70}")
                progress_log.append(f"✅ Run {run_num} COMPLETE!")
                progress_log.append(f"Selected {len(results['selected_variables'])} variables")
                display_text = "\n".join(progress_log[-30:])
                progress_log_placeholder.code(display_text, language=None)

            progress_text.text(f"✅ All {n_independent_runs} independent run{'s' if n_independent_runs != 1 else ''} completed!")
            progress_bar.progress(1.0)
            progress_log_placeholder.empty()  # Clear progress log

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: Calculate band consensus
            # ═══════════════════════════════════════════════════════════════

            st.subheader(f"📊 Band Consensus Analysis ({threshold_text})")

            # Create consensus matrix: rows=variables, cols=runs
            n_vars = X.shape[1]
            consensus_matrix = np.zeros((n_vars, n_independent_runs), dtype=int)

            for run_idx, selected_vars_set in enumerate(all_selected_variables):
                for var_idx in selected_vars_set:
                    if var_idx < n_vars:
                        consensus_matrix[var_idx, run_idx] = 1

            # Count selections for each variable
            consensus_frequency = np.sum(consensus_matrix, axis=1)

            # Apply dynamic consensus threshold (already calculated above)
            final_selected_variables = np.where(
                consensus_frequency >= consensus_threshold
            )[0].tolist()

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Variables", n_vars)
            with col2:
                st.metric("Runs Executed", n_independent_runs)
            with col3:
                st.metric("Consensus Threshold", f"≥{consensus_threshold}/{n_independent_runs}")
            with col4:
                st.metric("Final Selected", len(final_selected_variables))

            # ═══════════════════════════════════════════════════════════════
            # PHASE 3: Consensus summary table
            # ═══════════════════════════════════════════════════════════════

            # Create detailed consensus table (only for selected variables)
            consensus_data = []
            for var_idx in range(n_vars):
                if consensus_frequency[var_idx] >= consensus_threshold:
                    row = {'Variable': var_idx}

                    # Add columns for each run dynamically
                    for run_idx in range(n_independent_runs):
                        row[f'Run_{run_idx+1}'] = '✓' if consensus_matrix[var_idx, run_idx] else ''

                    row['Frequency'] = int(consensus_frequency[var_idx])
                    row['Status'] = '✅ SELECTED'
                    consensus_data.append(row)

            if consensus_data:
                consensus_df = pd.DataFrame(consensus_data)

                # Sort by frequency (descending)
                selected_vars_df = consensus_df.sort_values('Frequency', ascending=False)

                st.write(f"**Variables Selected in ≥{consensus_threshold}/{n_independent_runs} Runs:**")
                st.dataframe(
                    selected_vars_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("⚠️ No variables met the consensus threshold.")

            # Store in session state for later use
            st.session_state['consensus_matrix'] = consensus_matrix
            st.session_state['final_selected_variables'] = final_selected_variables
            st.session_state['consensus_frequency'] = consensus_frequency
            st.session_state['all_runs_results'] = all_runs_results
            st.session_state['all_selected_variables'] = all_selected_variables
            st.session_state['n_independent_runs'] = n_independent_runs
            st.session_state['consensus_threshold'] = consensus_threshold

            # Create a unified results dict for compatibility with existing code
            # Use the first run's results as base, but with consensus variables
            results = all_runs_results[0].copy()
            results['selected_variables'] = np.array(final_selected_variables)
            results['consensus_matrix'] = consensus_matrix
            results['consensus_frequency'] = consensus_frequency
            results['all_runs_results'] = all_runs_results

            # Store unified results
            st.session_state['gaplssp_results'] = results

            st.success("✅ Band consensus analysis complete!")

        except Exception as e:
            progress_placeholder = st.empty()
            progress_placeholder.error(f"❌ Error: {str(e)}")
            st.stop()

    # Display results
    if st.session_state['gaplssp_results'] is not None:
        results = st.session_state['gaplssp_results']

        st.subheader("📊 Results Summary")

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Selected Variables", len(results['selected_variables']))
        with col2:
            best_fitness = max(results['best_fitnesses']) if 'best_fitnesses' in results else results.get('best_fitness', 0)
            st.metric("Best Fitness", f"{best_fitness:.2f}%")
        with col3:
            best_comp = results.get('best_components', 'N/A')
            st.metric("Best Components", best_comp)

        # Tabs for detailed results
        tab1, tab2, tab3, tab4 = st.tabs([
            "Selection Frequency",
            "Fitness Evolution",
            "CV Curve",
            "Summary"
        ])

        # Tab 1: Selection Frequency
        with tab1:
            st.subheader("Variable Selection Frequency (Spectral Order)")

            # Check if consensus data is available
            if 'consensus_frequency' in st.session_state and 'final_selected_variables' in st.session_state:
                try:
                    # Get original feature names if available
                    feature_names = st.session_state.get('feature_names', None)

                    # Use new spectral order plot showing ALL variables
                    fig = plot_selection_frequency_spectral_order(
                        consensus_frequency=st.session_state['consensus_frequency'],
                        final_selected_variables=st.session_state['final_selected_variables'],
                        feature_names=feature_names,
                        title="Variable Selection Frequency (All Variables in Spectral Order)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Add explanation
                    st.info(f"**Green bars** = Selected (≥3/5 runs), **Blue bars** = Not selected. "
                           f"Shows all {X.shape[1]} variables in original spectral order.")

                except Exception as e:
                    st.error(f"Error plotting: {str(e)}")
            else:
                # Fallback to old plot if consensus data not available
                try:
                    fig = plot_selection_frequency(
                        results['selection_freq_smoothed'],
                        np.array(results['selected_variables']),
                        [f"Var {i}" for i in range(X.shape[1])],
                        top_n=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting: {str(e)}")

        # Tab 2: Fitness Evolution
        with tab2:
            st.subheader("Fitness Evolution Across Runs")
            try:
                fig = plot_fitness_evolution(
                    results['run_history'],
                    show_runs=10
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting: {str(e)}")

        # Tab 3: CV Curve
        with tab3:
            st.subheader("Cross-Validation Curve (Elbow Plot)")
            try:
                if results.get('stepwise_results'):
                    n_vars_list = [r['n_vars'] for r in results['stepwise_results']]
                    cv_list = [r['fitness'] for r in results['stepwise_results']]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=n_vars_list,
                        y=cv_list,
                        mode='lines+markers',
                        name='CV %',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=8)
                    ))

                    fig.update_layout(
                        title="CV% vs Number of Variables",
                        xaxis_title="Number of Variables",
                        yaxis_title="CV %",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No stepwise results available")
            except Exception as e:
                st.error(f"Error plotting: {str(e)}")

        # Tab 4: Summary Table
        with tab4:
            st.subheader("Stepwise Results")
            try:
                if results.get('stepwise_results'):
                    stepwise_df = pd.DataFrame(results['stepwise_results'])
                    st.dataframe(stepwise_df, use_container_width=True)
                else:
                    st.info("No stepwise results")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Selected variables detail
        st.subheader("📋 Selected Variables Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Variables ({len(results['selected_variables'])}):**")
            st.code(", ".join(map(str, sorted(results['selected_variables']))))

        with col2:
            st.write("**Selection Frequency (Top 10):**")
            top_10_idx = np.argsort(results['selection_freq_smoothed'])[-10:]
            for idx in reversed(top_10_idx):
                freq = results['selection_freq_smoothed'][idx]
                st.write(f"Var {idx}: {freq:.0f}x")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2.5: MULTI-RUN BAND SELECTION VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    if st.session_state['gaplssp_results'] is not None and 'all_selected_variables' in st.session_state:
        st.header("📊 Multi-Run Band Selection Pattern")

        st.markdown("""
        Visualize which variables were selected in each of the 5 independent runs:
        - **Red spectrum**: Mean absorption profile
        - **Colored tick marks**: Each run's selected variables (5 horizontal levels)
        - **Transparent bands**: Highlight regions selected by each run
        - **Overlap areas**: Variables selected by multiple runs (more robust)
        """)

        show_multirun = st.checkbox(
            "✅ Show multi-run selection pattern",
            value=True,
            key="show_multirun_bands"
        )

        if show_multirun:
            try:
                all_selected_variables = st.session_state['all_selected_variables']

                # Plot multi-run band selection
                fig = plot_multirun_band_selection(
                    X,
                    all_selected_variables,
                    title="Band Selection Pattern Across 5 Independent Runs"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add interpretation guide
                with st.expander("📖 How to interpret this plot"):
                    st.markdown("""
                    **What you're seeing:**
                    - Each of the 5 horizontal tick levels represents one independent GA run
                    - Vertical ticks show which variables that run selected
                    - Transparent colored bands show contiguous regions selected by each run
                    - Where multiple colors overlap = consensus (robust selection)

                    **What to look for:**
                    - **Dense overlap areas**: Variables consistently selected across runs → high confidence
                    - **Isolated ticks**: Variables selected by only 1-2 runs → less reliable
                    - **Spectral interpretation**: Do selected regions correspond to known chemical bands?

                    **Leardi philosophy:**
                    This visualization helps identify truly robust variable selections by showing
                    the variability across independent GA runs. Only variables selected in ≥3/5 runs
                    are used in the final consensus model.
                    """)

            except Exception as e:
                st.error(f"Error plotting multi-run visualization: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: SPECTRAL BANDS VISUALIZATION WITH CONSENSUS (OPTIONAL)
    # ═══════════════════════════════════════════════════════════════════════════

    if st.session_state['gaplssp_results'] is not None:
        st.header("🎨 Spectral Bands Visualization with Consensus")

        st.markdown("""
        Shows the full spectrum with consensus-selected bands highlighted:
        - **Red line**: Mean spectrum across all samples
        - **Green shaded regions**: Consensus bands (≥3/5 runs)
        - **Bottom ticks**: Individual run selections
        - Bands are contiguous regions of variables
        """)

        show_spectral = st.checkbox(
            "✅ Show spectral bands with consensus visualization",
            key="show_spectral_bands"
        )

        if show_spectral:
            try:
                results = st.session_state['gaplssp_results']

                # Check if consensus data is available
                if 'consensus_matrix' in results and 'final_selected_variables' in st.session_state:
                    final_selected_variables = st.session_state['final_selected_variables']
                    consensus_matrix = st.session_state['consensus_matrix']

                    # Plot spectrum with consensus bands
                    fig = plot_spectra_with_consensus_bands(
                        X,
                        final_selected_variables,
                        consensus_matrix,
                        title="Spectral Data with Consensus Bands (≥3/5 Runs)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Create bands table from consensus variables
                    bands_df = create_bands_table(
                        np.array(final_selected_variables),
                        [f"Var {i}" for i in range(X.shape[1])]
                    )

                    st.subheader(f"📋 Consensus Bands Summary ({len(bands_df)} total)")
                    st.dataframe(bands_df, use_container_width=True)

                else:
                    # Fallback to single-run visualization if consensus not available
                    selected_vars = np.array(results['selected_variables'])

                    fig = plot_spectrum_with_bands(
                        X,
                        selected_vars,
                        [f"Var {i}" for i in range(X.shape[1])],
                        title="Spectral Data with Selected Bands (Green Shaded)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    bands_df = create_bands_table(
                        selected_vars,
                        [f"Var {i}" for i in range(X.shape[1])]
                    )

                    st.subheader(f"Selected Bands ({len(bands_df)} total)")
                    st.dataframe(bands_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: EXPORT SELECTED DATASET
    # ═══════════════════════════════════════════════════════════════════════════

    if st.session_state['gaplssp_results'] is not None and 'final_selected_variables' in st.session_state:
        st.header("📤 Export Selected Dataset")

        st.markdown("""
        Export a new dataset containing **only the consensus-selected variables** to XLSX.

        **Structure matches ORIGINAL dataset:**
        - First column: Y (with **ORIGINAL column name**)
        - Following columns: Selected variables (with **ORIGINAL column names**)

        The file will include:
        - **Sheet 1 (Data)**: Y + selected X variables (ORIGINAL names!)
        - **Sheet 2 (Band Metadata)**: Variable indices, frequencies, run information
        - **Sheet 3 (Summary)**: Statistics, methodology, and column names
        """)

        # Check if we have the necessary data
        if len(st.session_state['final_selected_variables']) > 0:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.info(f"**{len(st.session_state['final_selected_variables'])}** variables will be exported "
                       f"(reduced from {X.shape[1]} = {(1 - len(st.session_state['final_selected_variables'])/X.shape[1]) * 100:.1f}% reduction)")

            with col2:
                # Generate timestamp for filename
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

                try:
                    # Get original column names from session state
                    y_col_name = st.session_state.get('y_column_name', 'Target')
                    feature_names = st.session_state.get('feature_names', None)

                    # Generate XLSX file with ORIGINAL column names
                    xlsx_bytes = export_selected_dataset(
                        X=X,
                        y=y,
                        final_selected_variables=st.session_state['final_selected_variables'],
                        consensus_matrix=st.session_state['consensus_matrix'],
                        consensus_frequency=st.session_state['consensus_frequency'],
                        feature_names=feature_names,
                        y_column_name=y_col_name,
                        dataset_name="selected_bands"
                    )

                    # Provide download button
                    st.download_button(
                        label="⬇️ Download XLSX",
                        data=xlsx_bytes,
                        file_name=f"selected_bands_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="primary"
                    )

                    st.success("✅ File ready for download!")

                except Exception as e:
                    st.error(f"❌ Export error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        else:
            st.warning("⚠️ No variables were selected. Run the consensus analysis first.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # FOOTER & DOCUMENTATION
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("""
    **References:**
    - Leardi R. et al. (2002)
    - "Variable selection for multivariate calibration using a genetic algorithm"
    - Analytica Chimica Acta, vol. 461, pp. 189-200
    """)

    st.caption(
        "Minimal implementation - GAPLSOPT + GAPLSSP only. "
        "No preset configurations, no alternative methods."
    )

    with st.expander("📚 How to Use", expanded=False):
        st.markdown("""
        **Step 1**: Dataset loaded ✅

        **Step 2**: Select variable & sample range
        - Choose which columns to include (first to last)
        - Choose which rows to include (first to last)

        **Step 3**: Select target variable
        - Choose which column is your target (y)
        - All other columns become features (X)

        **Step 4** (Optional): Run randomization test to find optimal evaluations
        - If you already know n_evaluations, you can skip this

        **Step 5** (Required): Run production GA for variable selection
        - Uses either recommended or custom number of evaluations

        **Outputs**:
        - Selection frequency: How often each variable was selected
        - Fitness evolution: Best fitness across runs
        - CV curve: Cross-validation % vs number of variables (find elbow)
        - Spectral bands: Visual representation of selected regions

        **Interpretation**:
        - Look for elbow point in CV curve
        - Select variables with highest frequency
        - For spectroscopic data, verify bands make chemical sense
        """)


# Call show() when run directly (not imported)
if __name__ == "__main__":
    show()
