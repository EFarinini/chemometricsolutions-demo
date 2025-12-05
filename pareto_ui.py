"""
Multi-Criteria Decision Making (Pareto Optimization) UI
Equivalent to DOE_Pareto.r
Complete Streamlit UI for Pareto front analysis with confidence intervals
"""

import streamlit as st
import pandas as pd
import numpy as np
from itertools import product

from .pareto_optimization import (
    calculate_pareto_front,
    plot_pareto_2d,
    plot_pareto_parallel_coordinates,
    plot_pareto_tradeoff_matrix,
    export_pareto_results,
    code_value,
    create_coding_dict,
    calculate_confidence_intervals_for_pareto
)
from .response_surface import create_prediction_matrix


def show_pareto_ui(model_results, x_vars, y_var):
    """
    Complete Pareto Optimization UI

    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        y_var: Y variable name
    """

    st.markdown("## 🎯 Multi-Criteria Decision Making (Pareto Optimization)")
    st.markdown("*Equivalent to DOE_Pareto.r*")

    st.info("""
    **Find optimal compromises** when you need to balance multiple objectives:
    - Maximize/minimize response(s) Y
    - Constrain predictor(s) X (e.g., minimize cost, target specific conditions)
    - Identify Pareto-optimal solutions (best tradeoffs)

    **When to use:** After fitting MLR model with good R-squared (>0.85), when you need to
    optimize multiple conflicting objectives simultaneously.
    """)

    # Check prerequisites
    if model_results is None:
        st.warning("⚠️ Fit an MLR model first in the 'Model Computation' tab")
        st.stop()

    # ========================================================================
    # SECTION 1: Grid Generation - User Defines Real Ranges
    # ========================================================================
    st.markdown("---")
    st.markdown("### 1️⃣ Grid Generation Settings")

    # CRITICAL EXPLANATION
    st.error("""
⚠️ **IMPORTANT: You Must Define Real Value Ranges**

Your training data is **coded** ([-1, +1]), but for practical use you need to specify what these coded values represent in **real units**.

**Why?** Because:
- Coded -1, 0, +1 have no physical meaning
- You need to interpret results (e.g., "optimal temperature = 75 degrees C" not "= 0.5 coded")
- You need to run actual experiments with real values

**What to enter:**
- The **REAL** minimum and maximum values that correspond to coded -1 and +1
- Example: If Coal_load_t_h coded range is [-1, +1], enter what -1 and +1 mean in reality (e.g., 5 t/h to 15 t/h)
    """)

    st.info("""
**Workflow:**
1. **You specify:** Real ranges (e.g., Temperature: 20-100 degrees C, Pressure: 1-5 bar)
2. **System generates:** Grid points in real units
3. **System codes:** Real to [-1, +1] for MLR prediction
4. **System predicts:** MLR computes response Y
5. **System displays:** Results in real units for interpretation

**Common unit examples:**
- Temperature: 20-100 degrees C, 50-200 degrees F
- Pressure: 1-5 bar, 10-50 psi
- Flow/Load: 5-15 t/h, 10-50 L/min
- Position/Opening: 0-100%
- Time: 0-60 min
- pH: 4-10
- Speed: 100-1000 RPM
    """)

    st.markdown("---")
    st.markdown("### Configure Real Ranges for Each Variable")

    st.warning(f"""
📋 **Your model has {len(x_vars)} predictor variable(s):** {', '.join(x_vars)}

For each variable below, you **must specify** the real min/max values that correspond to the coded range [-1, +1] used in your training data.
    """)

    # Grid configuration storage
    grid_config = {}

    # Create expander for each variable - USER MUST FILL!
    for idx, var in enumerate(x_vars, 1):
        with st.expander(f"🔧 Configure {var} (Variable {idx}/{len(x_vars)})", expanded=(idx == 1)):

            st.markdown(f"#### {var}")

            st.info(f"""
**What does coded -1 and +1 mean for {var}?**

In your training data, {var} is coded as [-1, 0, +1].
Now specify what these coded values represent in real/natural units.

**Example:**
- If {var} is temperature: coded -1 = 20 degrees C, coded +1 = 100 degrees C
- If {var} is pressure: coded -1 = 1 bar, coded +1 = 5 bar
- If {var} is position: coded -1 = 0%, coded +1 = 100%
            """)

            # Three columns for min, max, steps
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Real Minimum Value**")
                st.caption("What does coded **-1** represent?")

                real_min = st.number_input(
                    f"Min value for {var}:",
                    value=0.0,
                    key=f"pareto_realmin_{var}",
                    format="%.4f",
                    help=f"Enter the REAL value that corresponds to coded -1 for {var}"
                )

                st.caption("Example: 5 t/h, 20 degrees C, 0%, 1 bar")

            with col2:
                st.markdown("**Real Maximum Value**")
                st.caption("What does coded **+1** represent?")

                real_max = st.number_input(
                    f"Max value for {var}:",
                    value=100.0,
                    key=f"pareto_realmax_{var}",
                    format="%.4f",
                    help=f"Enter the REAL value that corresponds to coded +1 for {var}"
                )

                st.caption("Example: 15 t/h, 100 degrees C, 100%, 5 bar")

            with col3:
                st.markdown("**Grid Resolution**")
                st.caption("Number of levels")

                n_steps = st.number_input(
                    f"Steps:",
                    min_value=3,
                    max_value=50,
                    value=15,
                    key=f"pareto_steps_{var}",
                    help="Number of grid points (higher = finer resolution but more candidates)"
                )

                st.caption(f"Will generate {int(n_steps)} levels")

            # Validation
            if real_min >= real_max:
                st.error(f"❌ **Error:** Minimum ({real_min:.4f}) must be less than Maximum ({real_max:.4f})")
                st.stop()

            # Show the coding transformation
            real_center = (real_min + real_max) / 2
            real_range = (real_max - real_min) / 2

            st.success(f"""
✅ **Coding Transformation Defined:**

| Real Value | Coded Value | Calculation |
|------------|-------------|-------------|
| **{real_min:.4f}** (min) | **-1.00** | `(real - center) / range` |
| **{real_center:.4f}** (center) | **0.00** | `({real_center:.4f} - {real_center:.4f}) / {real_range:.4f}` |
| **{real_max:.4f}** (max) | **+1.00** | `({real_max:.4f} - {real_center:.4f}) / {real_range:.4f}` |

**Example real values that will be generated:**
{', '.join([f"{real_min + i*(real_max-real_min)/(int(n_steps)-1):.3f}" for i in range(min(5, int(n_steps)))])}{" ..." if int(n_steps) > 5 else ""}
            """)

            # Store configuration
            grid_config[var] = {
                'real_min': real_min,
                'real_max': real_max,
                'steps': int(n_steps)
            }

    # Summary table
    st.markdown("---")
    st.markdown("### 📋 Grid Configuration Summary")

    summary_data = []
    for var in x_vars:
        config = grid_config[var]
        real_center = (config['real_min'] + config['real_max']) / 2

        summary_data.append({
            'Variable': var,
            'Real Min (coded -1)': f"{config['real_min']:.4f}",
            'Real Center (coded 0)': f"{real_center:.4f}",
            'Real Max (coded +1)': f"{config['real_max']:.4f}",
            'Steps': config['steps']
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Calculate total candidates
    total_candidates = int(np.prod([config['steps'] for config in grid_config.values()]))

    if total_candidates > 100000:
        st.error(f"⚠️ **Too many candidates:** {total_candidates:,} points")
        st.error("Reduce step sizes to stay below 100,000 points (recommended: 10-20 steps per variable)")
        st.stop()
    else:
        st.info(f"📊 **Total candidate points to generate:** {total_candidates:,}")

        # Provide context
        if total_candidates < 100:
            st.caption("✅ Very fast generation (<1 second)")
        elif total_candidates < 1000:
            st.caption("✅ Fast generation (~1 second)")
        elif total_candidates < 10000:
            st.caption("⚠️ Moderate generation (~5 seconds)")
        else:
            st.caption("⚠️ Slower generation (~10-30 seconds)")

    # Generate Grid Button
    st.markdown("---")

    if st.button("🚀 Generate Candidate Grid", type="primary", key="gen_pareto_grid", use_container_width=True):
        with st.spinner(f"Generating {total_candidates:,} candidate points and computing MLR predictions..."):
            try:
                # STEP 1: Generate grid in REAL units
                st.text("Step 1/5: Generating grid in real units...")
                grid_ranges_real = []
                for var in x_vars:
                    config = grid_config[var]
                    real_range = np.linspace(config['real_min'],
                                            config['real_max'],
                                            config['steps'])
                    grid_ranges_real.append(real_range)

                grid_points_real = list(product(*grid_ranges_real))

                # STEP 2: Create DataFrame with REAL columns
                st.text("Step 2/5: Creating real-value DataFrame...")
                real_cols = [f'{v}_real' for v in x_vars]
                real_df = pd.DataFrame(grid_points_real, columns=real_cols)

                # STEP 3: CODE real → [-1, +1] for MLR
                st.text("Step 3/5: Coding real values to [-1, +1]...")
                coding_dict = create_coding_dict(x_vars, grid_config)
                coded_df = pd.DataFrame()

                for var in x_vars:
                    real_col = f'{var}_real'
                    coded_df[var] = real_df[real_col].apply(
                        lambda x: code_value(x,
                                           grid_config[var]['real_min'],
                                           grid_config[var]['real_max'])
                    )

                # STEP 4: PREDICT using MLR on CODED values
                st.text("Step 4/5: Computing MLR predictions...")
                coefficients = model_results['coefficients']
                coef_names = coefficients.index.tolist()

                X_model = create_prediction_matrix(
                    coded_df[x_vars].values,
                    x_vars,
                    coef_names
                )

                predictions = X_model @ coefficients.values

                # STEP 5: Combine everything
                st.text("Step 5/6: Finalizing candidate DataFrame...")
                candidate_df = pd.concat([coded_df, real_df], axis=1)
                candidate_df[f'{y_var}_predicted'] = predictions

                # STEP 6: Calculate Confidence Intervals
                st.text("Step 6/6: Calculating confidence intervals...")

                try:
                    candidate_df = calculate_confidence_intervals_for_pareto(
                        candidate_df,
                        model_results,
                        x_vars,
                        y_var,
                        confidence_level=0.95
                    )
                    ci_available = True
                except Exception as e:
                    st.warning(f"⚠️ Could not calculate confidence intervals: {str(e)}")
                    ci_available = False

                # Store in session state
                st.session_state.pareto_candidate_df = candidate_df
                st.session_state.pareto_coding_dict = coding_dict
                st.session_state.pareto_grid_config = grid_config
                st.session_state.pareto_ci_available = ci_available

                st.success(f"✅ **Successfully generated {len(candidate_df):,} candidate points!**")

                # Statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Grid Points", f"{len(candidate_df):,}")
                with col_stat2:
                    st.metric("Variables", len(x_vars))
                with col_stat3:
                    st.metric("Predictions", f"{len(predictions):,}")

                # Preview table (REAL units)
                st.markdown("---")
                st.markdown("#### 📋 Preview Generated Grid (First 10 Rows)")

                if ci_available:
                    st.info("""
**DataFrame structure:**
- Coded columns: Used for MLR predictions (hidden in preview)
- Real columns: Shown below for interpretation
- Predicted response: MLR output with 95% confidence intervals
                    """)
                else:
                    st.info("""
**DataFrame structure:**
- Coded columns: Used for MLR predictions (hidden in preview)
- Real columns: Shown below for interpretation
- Predicted response: MLR output
                    """)

                preview_cols = real_cols + [f'{y_var}_predicted']
                if ci_available:
                    preview_cols.extend([f'{y_var}_predicted_lower', f'{y_var}_predicted_upper'])

                st.dataframe(
                    candidate_df[preview_cols].head(10),
                    use_container_width=True,
                    hide_index=False
                )

                # Show prediction statistics
                st.markdown("#### 📊 Prediction Statistics")
                col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

                with col_pred1:
                    st.metric("Min Predicted Y", f"{predictions.min():.4f}")
                with col_pred2:
                    st.metric("Mean Predicted Y", f"{predictions.mean():.4f}")
                with col_pred3:
                    st.metric("Max Predicted Y", f"{predictions.max():.4f}")
                with col_pred4:
                    st.metric("Std Dev Y", f"{predictions.std():.4f}")

                st.rerun()

            except Exception as e:
                st.error(f"❌ **Grid generation failed:** {str(e)}")
                import traceback
                with st.expander("🐛 Error Details (Click to Expand)"):
                    st.code(traceback.format_exc())
                st.stop()

    # Check if grid exists
    if 'pareto_candidate_df' not in st.session_state:
        st.info("👆 **Next step:** Click 'Generate Candidate Grid' button above to create prediction grid")
        st.stop()

    # Show existing grid status
    candidate_df = st.session_state.pareto_candidate_df
    coding_dict = st.session_state.get('pareto_coding_dict', {})
    grid_config = st.session_state.get('pareto_grid_config', {})

    st.success(f"✅ **Grid Ready:** {len(candidate_df):,} candidate points with predictions")

    with st.expander("📊 View Current Grid Details"):
        st.markdown("**Grid Configuration:**")
        for var in x_vars:
            config = grid_config[var]
            st.write(f"- **{var}**: {config['real_min']:.4f} to {config['real_max']:.4f} ({config['steps']} steps)")

        st.markdown("**Preview Grid (Real Units):**")
        real_cols = [f'{v}_real' for v in x_vars]
        preview_cols = real_cols + [f'{y_var}_predicted']
        st.dataframe(candidate_df[preview_cols].head(20), use_container_width=True)

    # ====================================================================
    # SECTION 2: Objective Definition
    # ====================================================================
    _show_objective_definition(candidate_df, x_vars, y_var)

    # ====================================================================
    # SECTION 3: Run Pareto Analysis
    # ====================================================================
    _show_pareto_analysis(candidate_df, x_vars, y_var)


def _show_objective_definition(candidate_df, x_vars, y_var):
    """Section 2: Define optimization objectives"""

    st.markdown("---")
    st.markdown("### 2️⃣ Define Optimization Objectives")

    # Get CI availability flag
    ci_available = st.session_state.get('pareto_ci_available', False)

    if ci_available:
        st.info("""
**Select variables to optimize and their objectives:**
- **Maximize/Minimize**: Find highest/lowest values (e.g., yield, quality, cost, energy)
- **Target**: Get as close as possible to specific value
- **Confidence Intervals Available**: For response Y, you can optimize conservatively using CI bounds

**Example:** Maximize Y (yield) using lower CI bound (conservative) while minimizing X1 (temperature)
        """)
    else:
        st.info("""
**Select variables to optimize and their objectives:**
- **Maximize**: Find highest values (e.g., yield, quality)
- **Minimize**: Find lowest values (e.g., cost, energy, time)
- **Target**: Get as close as possible to specific value

**Example:** Maximize Y (yield) while minimizing X1 (temperature = energy cost)
        """)

    # Available columns for objectives
    # Use REAL columns for X variables, predicted for Y
    real_cols = [f'{v}_real' for v in x_vars]
    available_objectives = [f'{y_var}_predicted'] + real_cols

    # Objective builder
    objectives_dict = {}

    st.markdown("#### Select Objectives to Optimize")
    st.caption("Tip: Start with 2-3 objectives for clearer visualization")

    for obj in available_objectives:
        # Determine display name
        if obj == f'{y_var}_predicted':
            display_name = f"{y_var} (Response)"
        elif obj in real_cols:
            # Extract variable name from real column
            var_name = obj.replace('_real', '')
            display_name = f"{var_name} (Predictor - Real Units)"
        else:
            display_name = obj

        with st.expander(f"🎯 {display_name}"):
            include = st.checkbox(
                f"Include **{display_name}** in optimization",
                key=f"include_{obj}",
                help=f"Add {obj} as an optimization objective"
            )

            if include:
                # Special handling for response variable (Y) with CI
                is_response = (obj == f'{y_var}_predicted')

                if is_response and ci_available:
                    st.success("""
**Confidence Intervals Available!**

For the response variable, you can optimize considering prediction uncertainty.
This is more conservative and realistic than using point predictions only.
                    """)

                    col_obj1, col_obj2 = st.columns([1, 2])

                    with col_obj1:
                        obj_type = st.selectbox(
                            "Objective type:",
                            ["Maximize", "Minimize", "Target"],
                            key=f"objtype_{obj}",
                            help="Select optimization direction"
                        )

                    with col_obj2:
                        if obj_type == "Target":
                            default_target = float(candidate_df[obj].median())
                            target_val = st.number_input(
                                "Target value:",
                                value=default_target,
                                key=f"target_{obj}",
                                format="%.4f",
                                help="Desired value to approach"
                            )

                            # Option to use conservative estimate (target ± CI)
                            use_ci = st.checkbox(
                                "Use conservative estimate (penalize uncertainty)",
                                value=True,
                                key=f"use_ci_target_{obj}",
                                help="Penalize points with high prediction uncertainty"
                            )

                            if use_ci:
                                objectives_dict[obj] = ('target_ci', target_val)
                                st.caption(f"Will minimize: |Y - {target_val:.2f}| + CI width (more robust)")
                            else:
                                objectives_dict[obj] = ('target', target_val)
                                st.caption(f"Will minimize: |Y - {target_val:.2f}| (standard)")

                        else:  # Maximize or Minimize
                            use_ci = st.checkbox(
                                f"Use conservative estimate for {obj_type.lower()}",
                                value=True,
                                key=f"use_ci_{obj}",
                                help=f"For {obj_type.lower()}: use {'lower' if obj_type=='Maximize' else 'upper'} CI bound"
                            )

                            if use_ci:
                                if obj_type == "Maximize":
                                    objectives_dict[obj] = 'maximize_ci'  # Use lower bound
                                    st.caption("Will maximize **lower CI bound** (conservative)")
                                else:
                                    objectives_dict[obj] = 'minimize_ci'  # Use upper bound
                                    st.caption("Will minimize **upper CI bound** (conservative)")
                            else:
                                objectives_dict[obj] = obj_type.lower()
                                st.caption(f"Will {obj_type.lower()} **predicted value** (standard)")

                            # Show statistics
                            min_val = candidate_df[obj].min()
                            max_val = candidate_df[obj].max()
                            mean_val = candidate_df[obj].mean()
                            st.caption(f"Predicted range: {min_val:.3f} to {max_val:.3f} (mean: {mean_val:.3f})")

                else:
                    # Standard handling for predictors (X) - no CI, use REAL values
                    col_obj1, col_obj2 = st.columns([1, 2])

                    with col_obj1:
                        obj_type = st.selectbox(
                            "Objective:",
                            ["Maximize", "Minimize", "Target"],
                            key=f"objtype_{obj}",
                            help="Optimization direction"
                        )

                    with col_obj2:
                        if obj_type == "Target":
                            # Get reasonable default target
                            if obj in candidate_df.columns:
                                default_target = float(candidate_df[obj].median())
                            else:
                                default_target = 0.0

                            target_val = st.number_input(
                                "Target value:",
                                value=default_target,
                                key=f"target_{obj}",
                                format="%.4f",
                                help="Desired value to approach"
                            )
                            objectives_dict[obj] = ('target', target_val)
                            st.caption(f"Will minimize: |value - {target_val:.2f}|")
                        else:
                            objectives_dict[obj] = obj_type.lower()
                            st.caption(f"Will {obj_type.lower()} this variable")

                        # Show statistics
                        if obj in candidate_df.columns:
                            min_val = candidate_df[obj].min()
                            max_val = candidate_df[obj].max()
                            mean_val = candidate_df[obj].mean()
                            st.caption(f"Range: {min_val:.3f} to {max_val:.3f} (mean: {mean_val:.3f})")

    if len(objectives_dict) < 2:
        st.warning("⚠️ Select at least 2 objectives for multi-criteria optimization")
        st.info("Single-objective optimization can be solved with simple sorting")
        st.stop()

    # Display selected objectives
    st.markdown("#### 📋 Selected Objectives:")
    obj_summary = []
    for obj, obj_def in objectives_dict.items():
        # Get display name
        if obj == f'{y_var}_predicted':
            display_name = f"{y_var} (Response)"
        elif '_real' in obj:
            display_name = obj.replace('_real', ' (Real)')
        else:
            display_name = obj

        if isinstance(obj_def, tuple):
            if obj_def[0] == 'target':
                obj_summary.append(f"**{display_name}**: Target = {obj_def[1]:.3f}")
            elif obj_def[0] == 'target_ci':
                obj_summary.append(f"**{display_name}**: Target = {obj_def[1]:.3f} (with CI penalty)")
        elif obj_def == 'maximize_ci':
            obj_summary.append(f"**{display_name}**: Maximize (lower CI bound)")
        elif obj_def == 'minimize_ci':
            obj_summary.append(f"**{display_name}**: Minimize (upper CI bound)")
        else:
            obj_summary.append(f"**{display_name}**: {obj_def.capitalize()}")

    st.markdown(" • " + "  \n• ".join(obj_summary))

    # Store objectives in session state for next section
    st.session_state.pareto_objectives_dict = objectives_dict


def _show_pareto_analysis(candidate_df, x_vars, y_var):
    """Section 3: Run Pareto Analysis and show results"""

    # Check if objectives are defined
    if 'pareto_objectives_dict' not in st.session_state:
        return

    objectives_dict = st.session_state.pareto_objectives_dict

    st.markdown("---")
    st.markdown("### 3️⃣ Run Pareto Analysis")

    col_run1, col_run2, col_run3 = st.columns([2, 1, 1])

    with col_run1:
        n_fronts = st.slider(
            "Number of Pareto fronts to identify:",
            min_value=1,
            max_value=10,
            value=3,
            help="Front 1 = best compromises, Front 2 = second best, etc."
        )

    with col_run2:
        st.metric("📊 Candidates", f"{len(candidate_df):,}")

    with col_run3:
        st.metric("🎯 Objectives", len(objectives_dict))

    if st.button("🚀 Calculate Pareto Front", type="primary", key="run_pareto"):
        with st.spinner("Calculating Pareto fronts..."):
            try:
                # Run Pareto analysis
                pareto_results = calculate_pareto_front(
                    df=candidate_df,
                    objectives_dict=objectives_dict,
                    n_fronts=n_fronts
                )

                st.session_state.pareto_results = pareto_results
                st.session_state.pareto_objectives = objectives_dict
                st.success("✅ Pareto analysis complete!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Pareto analysis failed: {str(e)}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())

    # ====================================================================
    # SECTION 4: Display Results
    # ====================================================================
    if 'pareto_results' in st.session_state and st.session_state.pareto_results is not None:
        _show_pareto_results(x_vars, y_var)


def _show_pareto_results(x_vars, y_var):
    """Section 4: Display Pareto results with visualizations and export"""

    st.markdown("---")
    st.markdown("### 📊 Pareto Analysis Results")

    pareto_df = st.session_state.pareto_results
    objectives_dict = st.session_state.pareto_objectives

    # Summary metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    front1_count = (pareto_df['pareto_rank'] == 1).sum()
    front2_count = (pareto_df['pareto_rank'] == 2).sum()
    front3_count = (pareto_df['pareto_rank'] == 3).sum()
    dominated_count = pareto_df['is_dominated'].sum()

    with col_m1:
        st.metric("Total Candidates", f"{len(pareto_df):,}")
    with col_m2:
        st.metric("🏆 Front 1 (Best)", front1_count)
    with col_m3:
        st.metric("Front 2", front2_count)
    with col_m4:
        st.metric("Dominated", dominated_count)

    # Pareto Front Table
    st.markdown("#### 🏆 Pareto Front 1 (Optimal Compromises)")

    ci_available = st.session_state.get('pareto_ci_available', False)

    if ci_available:
        st.info("""
**Pareto Front 1** contains the best compromise solutions:
- No point is dominated by any other point - improving one objective would worsen another
- **Values shown in REAL units** for practical interpretation
- **Confidence intervals** included for predicted response (95% CI)
- **Crowding Distance** indicates diversity - higher values mean more unique solutions
        """)
    else:
        st.info("""
**Pareto Front 1** contains the best compromise solutions:
- No point is dominated by any other point - improving one objective would worsen another
- **Values shown in REAL units** for practical interpretation
- **Crowding Distance** indicates diversity - higher values mean more unique solutions
        """)

    front1_df = pareto_df[pareto_df['pareto_rank'] == 1].copy()

    # Sort by crowding distance (most diverse first)
    front1_df = front1_df.sort_values('crowding_distance', ascending=False)

    # Build display columns: REAL first, then CODED, then Y+CI, then metrics
    display_cols_ordered = []

    # 1. Add ALL X variables in REAL units first
    for var in x_vars:
        real_col = f'{var}_real'
        if real_col in front1_df.columns:
            display_cols_ordered.append(real_col)

    # 2. Add ALL X variables in CODED units (as reference)
    for var in x_vars:
        if var in front1_df.columns:
            display_cols_ordered.append(var)

    # 3. Add Y prediction
    display_cols_ordered.append(f'{y_var}_predicted')

    # 4. Add CI columns if available
    if ci_available:
        if f'{y_var}_predicted_lower' in front1_df.columns:
            display_cols_ordered.append(f'{y_var}_predicted_lower')
        if f'{y_var}_predicted_upper' in front1_df.columns:
            display_cols_ordered.append(f'{y_var}_predicted_upper')
        if f'{y_var}_ci_semiwidth' in front1_df.columns:
            display_cols_ordered.append(f'{y_var}_ci_semiwidth')

    # 5. Add Pareto metrics
    display_cols_ordered.extend(['pareto_rank', 'crowding_distance'])

    # Filter to only existing columns
    display_cols = [c for c in display_cols_ordered if c in front1_df.columns]

    # Build column config for better formatting
    column_config = {}

    # Format X variables (real)
    for var in x_vars:
        real_col = f'{var}_real'
        if real_col in display_cols:
            column_config[real_col] = st.column_config.NumberColumn(
                f'{var} (real)',
                format="%.4f"
            )

    # Format X variables (coded)
    for var in x_vars:
        if var in display_cols:
            column_config[var] = st.column_config.NumberColumn(
                f'{var} (coded)',
                format="%.4f"
            )

    # Format Y and CI columns
    column_config[f'{y_var}_predicted'] = st.column_config.NumberColumn('Y pred', format="%.4f")
    if ci_available:
        column_config[f'{y_var}_predicted_lower'] = st.column_config.NumberColumn('CI lower', format="%.4f")
        column_config[f'{y_var}_predicted_upper'] = st.column_config.NumberColumn('CI upper', format="%.4f")
        column_config[f'{y_var}_ci_semiwidth'] = st.column_config.NumberColumn('CI +/-', format="%.4f")

    st.dataframe(
        front1_df[display_cols].head(20),
        use_container_width=True,
        hide_index=False,
        column_config=column_config
    )

    # Download Front 1 as CSV with real values + CI
    csv_front1 = front1_df[display_cols].to_csv(index=True)
    csv_filename = f"Pareto_Front1_{y_var}_RealUnits_CI.csv" if ci_available else f"Pareto_Front1_{y_var}_RealUnits.csv"
    st.download_button(
        "💾 Download Pareto Front 1 (Real Units + CI)",
        csv_front1,
        csv_filename,
        "text/csv",
        key="download_front1_csv"
    )

    # ================================================================
    # SECTION 5: Visualizations
    # ================================================================
    _show_pareto_visualizations(pareto_df, front1_df, objectives_dict, y_var)

    # ================================================================
    # SECTION 6: Export
    # ================================================================
    _show_pareto_export(pareto_df, objectives_dict, y_var)


def _show_pareto_visualizations(pareto_df, front1_df, objectives_dict, y_var):
    """Section 5: Pareto visualizations"""

    st.markdown("---")
    st.markdown("### 📈 Pareto Front Visualizations")

    objective_list = list(objectives_dict.keys())
    ci_available = st.session_state.get('pareto_ci_available', False)

    # 2D Scatter plots (pairwise)
    if len(objective_list) >= 2:
        st.markdown("#### 2D Pareto Front Visualization")

        col_v1, col_v2, col_v3, col_v4 = st.columns([2, 2, 1, 1])

        with col_v1:
            obj1 = st.selectbox("X-axis objective:", objective_list, key="pareto_viz_x")
        with col_v2:
            available_y = [o for o in objective_list if o != obj1]
            obj2 = st.selectbox("Y-axis objective:", available_y, key="pareto_viz_y")
        with col_v3:
            show_arrows = st.checkbox("Show arrows", value=True, key="show_arrows")
        with col_v4:
            # Check if either axis involves Y response and CI is available
            y_pred_col = f'{y_var}_predicted'
            involves_response = (obj1 == y_pred_col or obj2 == y_pred_col)

            if ci_available and involves_response:
                use_ci_viz = st.checkbox(
                    "Show with CI",
                    value=False,
                    key="use_ci_visualization",
                    help="Use CI bounds instead of point predictions"
                )
            else:
                use_ci_viz = False

        # Prepare plot data - replace Y with CI bounds if requested
        plot_df = pareto_df.copy()

        if use_ci_viz and ci_available:
            st.info("📊 **Conservative view:** Using CI bounds instead of point predictions")

            # Replace Y_predicted with appropriate CI bound based on objective type
            if obj1 == y_pred_col:
                obj1_type = objectives_dict.get(obj1, 'maximize')
                if 'maximize' in str(obj1_type):
                    # For maximization, use lower bound (conservative)
                    if f'{y_var}_predicted_lower' in plot_df.columns:
                        plot_df[obj1] = plot_df[f'{y_var}_predicted_lower']
                        st.caption(f"X-axis: Using **lower CI bound** (conservative for maximize)")
                else:
                    # For minimization, use upper bound (conservative)
                    if f'{y_var}_predicted_upper' in plot_df.columns:
                        plot_df[obj1] = plot_df[f'{y_var}_predicted_upper']
                        st.caption(f"X-axis: Using **upper CI bound** (conservative for minimize)")

            if obj2 == y_pred_col:
                obj2_type = objectives_dict.get(obj2, 'maximize')
                if 'maximize' in str(obj2_type):
                    # For maximization, use lower bound (conservative)
                    if f'{y_var}_predicted_lower' in plot_df.columns:
                        plot_df[obj2] = plot_df[f'{y_var}_predicted_lower']
                        st.caption(f"Y-axis: Using **lower CI bound** (conservative for maximize)")
                else:
                    # For minimization, use upper bound (conservative)
                    if f'{y_var}_predicted_upper' in plot_df.columns:
                        plot_df[obj2] = plot_df[f'{y_var}_predicted_upper']
                        st.caption(f"Y-axis: Using **upper CI bound** (conservative for minimize)")

        fig_2d = plot_pareto_2d(
            plot_df, obj1, obj2, objectives_dict,
            color_by='pareto_rank', show_arrows=show_arrows
        )
        st.plotly_chart(fig_2d, use_container_width=True)

        st.caption("""
        Stars = Pareto Front 1 (optimal compromises)
        Circles = Other fronts
        Arrows show optimization direction
        """)

    # Parallel coordinates (for 3+ objectives)
    if len(objective_list) >= 3:
        st.markdown("---")
        st.markdown("#### Parallel Coordinates Plot")
        st.info("Shows relationships between all objectives for Pareto front points")

        n_display = st.slider(
            "Number of points to display:",
            min_value=10,
            max_value=min(100, len(front1_df)),
            value=min(50, len(front1_df)),
            key="parallel_n_points"
        )

        fig_parallel = plot_pareto_parallel_coordinates(
            front1_df,
            objective_list,
            n_points=n_display
        )
        st.plotly_chart(fig_parallel, use_container_width=True)

    # Tradeoff matrix
    if len(objective_list) >= 2 and len(front1_df) >= 5:
        st.markdown("---")
        st.markdown("#### Objective Tradeoff Matrix")
        st.info("Shows pairwise conflicts between objectives (Pareto Front 1 only)")

        fig_tradeoff = plot_pareto_tradeoff_matrix(front1_df, objectives_dict)
        if fig_tradeoff:
            st.plotly_chart(fig_tradeoff, use_container_width=True)


def _show_pareto_export(pareto_df, objectives_dict, y_var):
    """Section 6: Export Pareto results"""

    st.markdown("---")
    st.markdown("### 💾 Export Pareto Results")

    st.info("""
    Export complete Pareto analysis to Excel with multiple sheets:
    - **Objectives**: List of optimization objectives
    - **Statistics**: Summary metrics
    - **Front_1, Front_2, Front_3**: Pareto fronts with all details
    - **All_Points**: Complete candidate set with rankings
    """)

    if st.button("📦 Export Pareto Analysis to Excel", type="primary", key="export_pareto"):
        with st.spinner("Creating Excel file..."):
            try:
                excel_buffer = export_pareto_results(
                    pareto_df,
                    objectives_dict,
                    filename=f'Pareto_Analysis_{y_var}.xlsx'
                )

                st.success("✅ Excel file ready for download!")

                st.download_button(
                    "📄 Download Pareto Analysis (XLSX)",
                    excel_buffer.getvalue(),
                    f"Pareto_Analysis_{y_var}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_pareto_excel"
                )

            except Exception as e:
                st.error(f"❌ Excel export failed: {str(e)}")
                st.info("CSV export of Front 1 is still available above")
