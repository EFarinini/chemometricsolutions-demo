"""
DoE Design Generators
Implements various experimental design generation methods.
"""

import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Optional


# ============================================================================
# VERIFIED PLACKETT-BURMANN DESIGNS (Hardcoded from EXC_PB.xlsx)
# ============================================================================

PB_DESIGNS = {
    8: [
        [1, -1, -1, 1, -1, 1, 1],
        [1, 1, -1, -1, 1, -1, 1],
        [1, 1, 1, -1, -1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1],
        [1, -1, 1, 1, 1, -1, -1],
        [-1, 1, -1, 1, 1, 1, -1],
        [-1, -1, 1, -1, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    12: [
        [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
        [-1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
        [1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1],
        [-1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1],
        [-1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1],
        [-1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1],
        [1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1],
        [1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1],
        [1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    16: [
        [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
        [-1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1],
        [-1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1],
        [-1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1],
        [-1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1],
        [1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1],
        [-1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1],
        [1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1],
        [-1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
}


# ============================================================================
# FRACTIONAL FACTORIAL DESIGNS (Hardcoded from EXC_FFFF.xlsx)
# ============================================================================

FF_DESIGNS = {
    # 2^4-1: 8 runs (resolution IV, 4 factors)
    "2^4-1": {
        "runs": 8,
        "factors": 4,
        "resolution": "IV",
        "matrix": [
            [-1, -1, -1, -1],
            [1, -1, -1, 1],
            [-1, 1, -1, 1],
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
            [1, -1, 1, -1],
            [-1, 1, 1, -1],
            [1, 1, 1, 1],
        ]
    },
    # 2^5-1: 16 runs (resolution V, 5 factors)
    "2^5-1": {
        "runs": 16,
        "factors": 5,
        "resolution": "V",
        "matrix": [
            [-1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1],
            [-1, 1, -1, -1, -1],
            [1, 1, -1, -1, 1],
            [-1, -1, 1, -1, -1],
            [1, -1, 1, -1, 1],
            [-1, 1, 1, -1, 1],
            [1, 1, 1, -1, -1],
            [-1, -1, -1, 1, -1],
            [1, -1, -1, 1, 1],
            [-1, 1, -1, 1, 1],
            [1, 1, -1, 1, -1],
            [-1, -1, 1, 1, 1],
            [1, -1, 1, 1, -1],
            [-1, 1, 1, 1, -1],
            [1, 1, 1, 1, 1],
        ]
    }
}


def get_design_info() -> Dict:
    """
    Return complete design specifications for all available DoE types.

    Returns:
        Dictionary with design metadata including formulas, requirements, and use cases
    """
    return {
        'full_factorial': {
            'name': 'Full Factorial',
            'n_experiments_formula': lambda n_vars, n_levels: n_levels ** n_vars,
            'min_variables': 1,
            'max_variables': 10,
            'levels': 'any',
            'resolution': 'full',
            'coding': 'actual levels',
            'use_cases': [
                'Complete exploration of factor space',
                'All main effects and interactions estimable',
                'Best for 2-7 factors with 2-3 levels each'
            ],
            'description': 'Tests all possible combinations of factor levels'
        },
        'fractional_factorial_IV': {
            'name': 'Fractional Factorial (Resolution IV)',
            'n_experiments_formula': lambda n_vars, n_levels=2: 2 ** (n_vars - 1),
            'min_variables': 3,
            'max_variables': 15,
            'levels': 2,
            'resolution': 'IV',
            'coding': 'coded (-1, +1)',
            'use_cases': [
                'Efficient screening when resources limited',
                'Main effects clear, two-factor interactions partially confounded',
                'Ideal for 5-15 factors at 2 levels'
            ],
            'description': 'Half-fraction design: main effects not confounded with each other'
        },
        'fractional_factorial_V': {
            'name': 'Fractional Factorial (Resolution V)',
            'n_experiments_formula': lambda n_vars, n_levels=2: 2 ** (n_vars - 2) if n_vars >= 5 else 2 ** n_vars,
            'min_variables': 5,
            'max_variables': 20,
            'levels': 2,
            'resolution': 'V',
            'coding': 'coded (-1, +1)',
            'use_cases': [
                'Main effects + two-factor interactions estimable',
                'Higher resolution than IV, fewer runs than full factorial',
                'Best for 6-12 factors when interactions matter'
            ],
            'description': 'Quarter-fraction design: main effects and 2FI not confounded'
        },
        'plackett_burman': {
            'name': 'Plackett-Burman',
            'n_experiments_formula': lambda n_vars: ((n_vars - 1) // 4 + 1) * 4,
            'min_variables': 2,
            'max_variables': 100,
            'levels': 2,
            'resolution': 'III',
            'coding': 'coded (-1, +1)',
            'use_cases': [
                'Highly efficient screening (n+1 runs for n factors)',
                'Main effects estimable only',
                'Ideal for initial screening with many factors'
            ],
            'description': 'Ultra-efficient screening design for main effects only'
        },
        'central_composite': {
            'name': 'Central Composite Design (CCD)',
            'n_experiments_formula': lambda n_vars: 2**n_vars + 2*n_vars + 5,
            'min_variables': 2,
            'max_variables': 10,
            'levels': 'continuous (5 levels: -α, -1, 0, +1, +α)',
            'resolution': 'full',
            'coding': 'coded with axial points',
            'use_cases': [
                'Response Surface Methodology (RSM)',
                'Fit quadratic models for optimization',
                'Best for 2-5 continuous factors'
            ],
            'description': 'Factorial + axial + center points for quadratic modeling'
        }
    }


def validate_design_feasibility(design_type: str, n_variables: int, n_levels: int = 2) -> Dict:
    """
    Check if design is feasible given number of variables.

    Args:
        design_type: Type of design
        n_variables: Number of factors
        n_levels: Number of levels (default 2)

    Returns:
        Dictionary with 'feasible' (bool), 'message' (str), 'n_experiments' (int)
    """
    designs = get_design_info()

    if design_type not in designs:
        return {
            'feasible': False,
            'message': f"Unknown design type: {design_type}",
            'n_experiments': 0
        }

    design = designs[design_type]

    # Check variable count constraints
    if n_variables < design['min_variables']:
        return {
            'feasible': False,
            'message': f"{design['name']} requires at least {design['min_variables']} variables (you have {n_variables})",
            'n_experiments': 0
        }

    if n_variables > design['max_variables']:
        return {
            'feasible': False,
            'message': f"{design['name']} supports max {design['max_variables']} variables (you have {n_variables})",
            'n_experiments': 0
        }

    # Check level constraints
    if design['levels'] == 2 and n_levels != 2:
        return {
            'feasible': False,
            'message': f"{design['name']} requires exactly 2 levels per variable",
            'n_experiments': 0
        }

    # Calculate number of experiments
    try:
        if design_type == 'full_factorial':
            n_exp = design['n_experiments_formula'](n_variables, n_levels)
        elif design_type in ['fractional_factorial_IV', 'fractional_factorial_V', 'central_composite']:
            n_exp = design['n_experiments_formula'](n_variables)
        elif design_type == 'plackett_burman':
            n_exp = design['n_experiments_formula'](n_variables)
        else:
            n_exp = 0

        # Sanity check: warn if too many experiments
        if n_exp > 10000:
            return {
                'feasible': False,
                'message': f"Design would require {n_exp:,} experiments (too large). Consider fractional design.",
                'n_experiments': n_exp
            }

        return {
            'feasible': True,
            'message': f"✓ {design['name']}: {n_exp} experiments required",
            'n_experiments': n_exp
        }

    except Exception as e:
        return {
            'feasible': False,
            'message': f"Error calculating design size: {str(e)}",
            'n_experiments': 0
        }


def get_design_description(design_type: str, n_variables: int, n_levels: int = 2) -> str:
    """
    Get human-readable design description.

    Args:
        design_type: Type of design
        n_variables: Number of factors
        n_levels: Number of levels

    Returns:
        Formatted description string
    """
    designs = get_design_info()

    if design_type not in designs:
        return f"Unknown design: {design_type}"

    design = designs[design_type]
    validation = validate_design_feasibility(design_type, n_variables, n_levels)

    description = f"""
**{design['name']}**

**Description:** {design['description']}

**Configuration:**
- Variables: {n_variables}
- Levels: {design['levels']}
- Resolution: {design['resolution']}
- Coding: {design['coding']}

**Experiments Required:** {validation['n_experiments']}

**Use Cases:**
"""

    for use_case in design['use_cases']:
        description += f"\n- {use_case}"

    return description


def generate_full_factorial(variables_config: dict) -> pd.DataFrame:
    """
    Generate full factorial design with all combinations of factor levels.

    Args:
        variables_config: Dictionary with variable configurations
            Format: {
                'Variable1': {'min': 0, 'max': 100, 'levels': [0, 50, 100]},
                'Variable2': {'min': 20, 'max': 80, 'levels': [20, 50, 80]},
            }

    Returns:
        DataFrame with all combinations (rows = n_levels^n_vars)
        Metadata stored in df.attrs
    """
    # Extract variable names in order
    var_names = list(variables_config.keys())

    # Extract levels for each variable
    levels_list = []
    for var_name in var_names:
        if 'levels' in variables_config[var_name]:
            levels = variables_config[var_name]['levels']
        else:
            # Generate levels from min, max, and step
            min_val = variables_config[var_name]['min']
            max_val = variables_config[var_name]['max']
            step = variables_config[var_name].get('step', (max_val - min_val) / 2)
            levels = list(np.arange(min_val, max_val + step/2, step))

        levels_list.append(levels)

    # Generate all combinations using itertools.product
    combinations = list(itertools.product(*levels_list))

    # Create DataFrame
    df = pd.DataFrame(combinations, columns=var_names)

    # Add experiment ID
    df.insert(0, 'Experiment_ID', range(1, len(df) + 1))

    # ═══════════════════════════════════════════════════════════════════════════
    # ADD METADATA
    # ═══════════════════════════════════════════════════════════════════════════

    n_vars = len(var_names)
    n_levels = len(levels_list[0]) if levels_list else 2

    df.attrs['design_type'] = 'full_factorial'
    df.attrs['design_name'] = 'Full Factorial'
    df.attrs['n_variables'] = n_vars
    df.attrs['n_levels'] = n_levels
    df.attrs['n_experiments'] = len(df)
    df.attrs['resolution'] = 'full'
    df.attrs['coding'] = 'actual levels'
    df.attrs['description'] = f'Full factorial design: {n_vars} variables, {n_levels} levels, {len(df)} experiments'
    df.attrs['auto_adjusted'] = False  # No auto-adjustment for full factorial

    return df


def generate_fractional_factorial_IV(variables_config: dict) -> pd.DataFrame:
    """
    Generate Fractional Factorial 2^(k-1) Resolution IV using VERIFIED designs.

    Args:
        variables_config: Dictionary with variable configurations

    Returns:
        DataFrame with fractional factorial design (Resolution IV)
    """
    var_names = list(variables_config.keys())
    n_vars = len(var_names)

    # Check if we have design for this size
    design_key = f"2^{n_vars}-1"

    if design_key not in FF_DESIGNS:
        raise ValueError(f"Fractional Factorial {design_key} not available. "
                        f"Available: {list(FF_DESIGNS.keys())}")

    design_spec = FF_DESIGNS[design_key]
    design_coded = design_spec['matrix']

    # Extract levels (min/max)
    levels_list = []
    for var_name in var_names:
        config = variables_config[var_name]

        if 'levels' in config and len(config['levels']) > 0:
            levels = [min(config['levels']), max(config['levels'])]
        else:
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            levels = [min_val, max_val]

        levels_list.append(levels)

    # Convert coded design to actual values
    design_actual = []
    for row_coded in design_coded:
        row_actual = []
        for i, code in enumerate(row_coded):
            # Map -1 to min, +1 to max
            value = levels_list[i][0] if code == -1 else levels_list[i][1]
            row_actual.append(value)
        design_actual.append(row_actual)

    df = pd.DataFrame(design_actual, columns=var_names)
    df.insert(0, 'Experiment_ID', range(1, len(df) + 1))

    # ═══════════════════════════════════════════════════════════════════════════
    # ADD METADATA
    # ═══════════════════════════════════════════════════════════════════════════

    df.attrs['design_type'] = 'fractional_factorial_IV'
    df.attrs['design_name'] = f'Fractional Factorial {design_key} Resolution IV'
    df.attrs['n_experiments'] = len(df)
    df.attrs['n_variables'] = n_vars
    df.attrs['n_levels'] = 2
    df.attrs['resolution'] = 'IV'
    df.attrs['coding'] = '[-1, +1]'
    df.attrs['description'] = f'Fractional Factorial {design_key}: {len(df)} experiments, Resolution IV'
    df.attrs['auto_adjusted'] = True
    df.attrs['adjustment_note'] = 'Multi-level variables → 2 levels (min/max)'
    df.attrs['verified'] = True

    return df


def generate_fractional_factorial_V(variables_config: dict) -> pd.DataFrame:
    """
    Generate Fractional Factorial 2^(k-2) Resolution V using VERIFIED designs.

    Args:
        variables_config: Dictionary with variable configurations

    Returns:
        DataFrame with fractional factorial design (Resolution V)
    """
    var_names = list(variables_config.keys())
    n_vars = len(var_names)

    # Check if we have design for this size
    design_key = f"2^{n_vars}-1"  # Note: 2^5-1 is actually Resolution V

    if design_key not in FF_DESIGNS:
        raise ValueError(f"Fractional Factorial {design_key} not available. "
                        f"Available: {list(FF_DESIGNS.keys())}")

    design_spec = FF_DESIGNS[design_key]

    # Verify it's Resolution V
    if design_spec['resolution'] != 'V':
        raise ValueError(f"Design {design_key} is Resolution {design_spec['resolution']}, not V")

    design_coded = design_spec['matrix']

    # Extract levels (min/max)
    levels_list = []
    for var_name in var_names:
        config = variables_config[var_name]

        if 'levels' in config and len(config['levels']) > 0:
            levels = [min(config['levels']), max(config['levels'])]
        else:
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            levels = [min_val, max_val]

        levels_list.append(levels)

    # Convert coded design to actual values
    design_actual = []
    for row_coded in design_coded:
        row_actual = []
        for i, code in enumerate(row_coded):
            # Map -1 to min, +1 to max
            value = levels_list[i][0] if code == -1 else levels_list[i][1]
            row_actual.append(value)
        design_actual.append(row_actual)

    df = pd.DataFrame(design_actual, columns=var_names)
    df.insert(0, 'Experiment_ID', range(1, len(df) + 1))

    # ═══════════════════════════════════════════════════════════════════════════
    # ADD METADATA
    # ═══════════════════════════════════════════════════════════════════════════

    df.attrs['design_type'] = 'fractional_factorial_V'
    df.attrs['design_name'] = f'Fractional Factorial {design_key} Resolution V'
    df.attrs['n_experiments'] = len(df)
    df.attrs['n_variables'] = n_vars
    df.attrs['n_levels'] = 2
    df.attrs['resolution'] = 'V'
    df.attrs['coding'] = '[-1, +1]'
    df.attrs['description'] = f'Fractional Factorial {design_key}: {len(df)} experiments, Resolution V'
    df.attrs['auto_adjusted'] = True
    df.attrs['adjustment_note'] = 'Multi-level variables → 2 levels (min/max)'
    df.attrs['verified'] = True

    return df


def generate_plackett_burman(variables_config: dict) -> pd.DataFrame:
    """
    Generate Plackett-Burman screening design using VERIFIED designs.

    Args:
        variables_config: Dictionary with variable configurations

    Returns:
        DataFrame with Plackett-Burman design
    """
    var_names = list(variables_config.keys())
    n_vars = len(var_names)

    # Find appropriate PB design (must have at least n_vars columns)
    valid_sizes = sorted([k for k in PB_DESIGNS.keys()])
    pb_size = None
    for size in valid_sizes:
        if size >= n_vars + 1:  # +1 for constraint column
            pb_size = size
            break

    if pb_size is None:
        raise ValueError(f"Cannot generate Plackett-Burman for {n_vars} variables. "
                        f"Max available: {max(valid_sizes)-1} variables")

    # Get hardcoded PB design
    design_coded = PB_DESIGNS[pb_size]

    # Take only n_vars columns
    design_coded = [row[:n_vars] for row in design_coded]

    # Extract levels (min/max)
    levels_list = []
    for var_name in var_names:
        config = variables_config[var_name]

        if 'levels' in config and len(config['levels']) > 0:
            levels = [min(config['levels']), max(config['levels'])]
        else:
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            levels = [min_val, max_val]

        levels_list.append(levels)

    # Convert coded design to actual values
    design_actual = []
    for row_coded in design_coded:
        row_actual = []
        for i, code in enumerate(row_coded):
            # Map -1 to min, +1 to max
            value = levels_list[i][0] if code == -1 else levels_list[i][1]
            row_actual.append(value)
        design_actual.append(row_actual)

    df = pd.DataFrame(design_actual, columns=var_names)
    df.insert(0, 'Experiment_ID', range(1, len(df) + 1))

    # ═══════════════════════════════════════════════════════════════════════════
    # ADD METADATA
    # ═══════════════════════════════════════════════════════════════════════════

    df.attrs['design_type'] = 'plackett_burman'
    df.attrs['design_name'] = f'Plackett-Burman N={pb_size}'
    df.attrs['n_experiments'] = len(df)
    df.attrs['n_variables'] = n_vars
    df.attrs['n_levels'] = 2
    df.attrs['resolution'] = 'Screening'
    df.attrs['coding'] = '[-1, +1]'
    df.attrs['pb_size'] = pb_size
    df.attrs['description'] = f'Plackett-Burman N={pb_size}: {n_vars} variables, {len(df)} runs'
    df.attrs['auto_adjusted'] = (pb_size != (n_vars + 1))
    df.attrs['verified'] = True

    return df


def generate_central_composite(variables_config: dict, alpha_type: str = 'orthogonal') -> pd.DataFrame:
    """
    Generate Central Composite Design (CCD) for Response Surface Methodology.

    Args:
        variables_config: Dictionary with variable configurations
        alpha_type: Type of alpha ('orthogonal', 'rotatable', 'face')

    Returns:
        DataFrame with CCD points (factorial + axial + center)
    """
    var_names = list(variables_config.keys())
    n_vars = len(var_names)

    try:
        from pyDOE2 import ccdesign

        # Generate CCD in coded form
        if alpha_type == 'orthogonal':
            design_coded = ccdesign(n_vars, center=(0, 0), alpha='o')
        elif alpha_type == 'rotatable':
            design_coded = ccdesign(n_vars, center=(0, 0), alpha='r')
        elif alpha_type == 'face':
            design_coded = ccdesign(n_vars, center=(0, 0), alpha='f')
        else:
            raise ValueError(f"Invalid alpha_type: {alpha_type}")

        # Add center points (typically 3-5 replicates)
        n_center = 5
        center_points = np.zeros((n_center, n_vars))
        design_coded = np.vstack([design_coded, center_points])

        # Convert to actual levels
        design_actual = np.zeros_like(design_coded, dtype=float)
        for i, var_name in enumerate(var_names):
            if 'levels' in variables_config[var_name]:
                levels = variables_config[var_name]['levels']
                min_val, max_val = min(levels), max(levels)
            else:
                min_val = variables_config[var_name]['min']
                max_val = variables_config[var_name]['max']

            center = (max_val + min_val) / 2
            half_range = (max_val - min_val) / 2

            # Map coded values to actual values
            design_actual[:, i] = center + design_coded[:, i] * half_range

        df = pd.DataFrame(design_actual, columns=var_names)

    except ImportError:
        # Fallback: manual CCD generation
        # Factorial points (2^k)
        factorial_coded = np.array(list(itertools.product([-1, 1], repeat=n_vars)))

        # Axial points (2*k)
        axial_coded = []
        for i in range(n_vars):
            point_pos = [0] * n_vars
            point_neg = [0] * n_vars

            # Calculate alpha
            if alpha_type == 'rotatable':
                alpha = n_vars ** 0.25
            elif alpha_type == 'face':
                alpha = 1
            else:  # orthogonal
                alpha = (2 ** n_vars) ** 0.25

            point_pos[i] = alpha
            point_neg[i] = -alpha
            axial_coded.append(point_pos)
            axial_coded.append(point_neg)

        axial_coded = np.array(axial_coded)

        # Center points
        n_center = 5
        center_coded = np.zeros((n_center, n_vars))

        # Combine all points
        design_coded = np.vstack([factorial_coded, axial_coded, center_coded])

        # Convert to actual levels
        design_actual = np.zeros_like(design_coded, dtype=float)
        for i, var_name in enumerate(var_names):
            if 'levels' in variables_config[var_name]:
                levels = variables_config[var_name]['levels']
                min_val, max_val = min(levels), max(levels)
            else:
                min_val = variables_config[var_name]['min']
                max_val = variables_config[var_name]['max']

            center = (max_val + min_val) / 2
            half_range = (max_val - min_val) / 2

            design_actual[:, i] = center + design_coded[:, i] * half_range

        df = pd.DataFrame(design_actual, columns=var_names)

    # Add experiment ID
    df.insert(0, 'Experiment_ID', range(1, len(df) + 1))

    # ═══════════════════════════════════════════════════════════════════════════
    # ADD METADATA
    # ═══════════════════════════════════════════════════════════════════════════

    # Calculate expected number of runs
    expected_runs = 2**n_vars + 2*n_vars + 5
    actual_runs = len(df)
    auto_adjusted = (actual_runs != expected_runs)

    df.attrs['design_type'] = 'central_composite'
    df.attrs['design_name'] = 'Central Composite Design (CCD)'
    df.attrs['n_variables'] = n_vars
    df.attrs['n_levels'] = '5 (continuous: -α, -1, 0, +1, +α)'
    df.attrs['n_experiments'] = len(df)
    df.attrs['resolution'] = 'full'
    df.attrs['alpha_type'] = alpha_type
    df.attrs['coding'] = f'coded with {alpha_type} alpha, then decoded to actual levels'
    df.attrs['description'] = f'Central Composite Design ({alpha_type}): {n_vars} variables, {len(df)} runs (factorial + axial + center)'
    df.attrs['auto_adjusted'] = auto_adjusted
    if auto_adjusted:
        df.attrs['adjustment_note'] = f'Design auto-adjusted from {expected_runs} to {actual_runs} runs (center point replication)'

    return df


# ============================================================================
# REAL/CODED MATRIX CONVERSION FUNCTIONS
# ============================================================================

def create_real_and_coded_matrices(df_real: pd.DataFrame, variables_config: dict) -> pd.DataFrame:
    """
    Convert real values to coded values ([-1, +1] or [-1, 0, +1]).

    Coding transformation:
    - For 2-level designs (min, max):
        coded = 2 * (real - min) / (max - min) - 1
        → min maps to -1, max maps to +1

    - For 3-level designs (min, center, max):
        coded = (real - center) / half_range
        → min maps to -1, center maps to 0, max maps to +1

    Args:
        df_real: DataFrame with real values (columns: Experiment_ID, X1, X2, ...)
        variables_config: Dictionary with variable configurations
            Format: {
                'X1': {'min': 10, 'max': 90, 'levels': [10, 90]},
                'X2': {'min': 20, 'max': 80, 'levels': [20, 50, 80]},
            }

    Returns:
        DataFrame with coded values (same structure as df_real)
    """
    df_coded = df_real.copy()

    # Get variable names (exclude Experiment_ID)
    var_names = [col for col in df_real.columns if col != 'Experiment_ID']

    for var_name in var_names:
        if var_name not in variables_config:
            continue

        config = variables_config[var_name]

        # Get min/max/center from config
        if 'levels' in config and len(config['levels']) > 0:
            levels = sorted(config['levels'])
            min_val = levels[0]
            max_val = levels[-1]
            center_val = levels[len(levels) // 2] if len(levels) == 3 else None
        else:
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            center_val = config.get('center', None)

        # Transform to coded values
        real_values = df_real[var_name].values

        if center_val is not None:
            # 3-level design: use center-based coding
            half_range = (max_val - min_val) / 2
            coded_values = (real_values - center_val) / half_range
        else:
            # 2-level design: linear scaling to [-1, +1]
            coded_values = 2 * (real_values - min_val) / (max_val - min_val) - 1

        df_coded[var_name] = coded_values

    return df_coded


def prepare_download_dataframe(df_real: pd.DataFrame, df_coded: pd.DataFrame) -> pd.DataFrame:
    """
    Combine real and coded matrices into interleaved format for download.

    Output format:
    Exp_ID | X1_Real | X1_Coded | X2_Real | X2_Coded | X3_Real | X3_Coded | ...

    Args:
        df_real: DataFrame with real values
        df_coded: DataFrame with coded values

    Returns:
        Combined DataFrame with interleaved Real/Coded columns
    """
    # Get variable names (exclude Experiment_ID)
    var_names = [col for col in df_real.columns if col != 'Experiment_ID']

    # Create new DataFrame starting with Experiment_ID
    combined_data = {'Experiment_ID': df_real['Experiment_ID'].values}

    # Interleave Real and Coded columns
    for var_name in var_names:
        combined_data[f'{var_name}_Real'] = df_real[var_name].values
        combined_data[f'{var_name}_Coded'] = df_coded[var_name].values

    df_combined = pd.DataFrame(combined_data)

    return df_combined


def create_combined_csv_side_by_side(df_real: pd.DataFrame, df_coded: pd.DataFrame) -> str:
    """
    Create CSV with Real and Coded values SIDE-BY-SIDE in one table.

    Format (no comments, no special chars):
    Experiment_ID,X1,X2,X3,...,X1,X2,X3,...
    1,0,0,0,...,-1,-1,-1,...
    2,0,0,100,...,-1,-1,1,...

    Args:
        df_real: DataFrame with real values
        df_coded: DataFrame with coded values

    Returns:
        String with CSV content
    """
    # Get variable names (exclude Experiment_ID)
    var_names = [col for col in df_real.columns if col != 'Experiment_ID']

    # Build header: Experiment_ID, [Real vars], [Coded vars]
    header = ['Experiment_ID']
    header.extend(var_names)  # X1, X2, X3, ...
    header.extend(var_names)  # X1, X2, X3, ... (again for coded)

    # Build data rows
    rows = []
    for idx in range(len(df_real)):
        row = [str(int(df_real.loc[idx, 'Experiment_ID']))]

        # Real values
        for var_name in var_names:
            val = df_real.loc[idx, var_name]
            row.append(str(int(val) if val == int(val) else val))

        # Coded values
        for var_name in var_names:
            val = df_coded.loc[idx, var_name]
            row.append(str(int(val)))

        rows.append(','.join(row))

    # Combine: header + data
    csv_content = [','.join(header)]
    csv_content.extend(rows)

    return '\n'.join(csv_content)


def generate_custom_design(variables_config: dict, design_type: str) -> pd.DataFrame:
    """
    Dispatcher function that calls appropriate design generator with validation.

    Args:
        variables_config: Dictionary with variable configurations
        design_type: Type of design ('full_factorial', 'plackett_burman', 'central_composite')

    Returns:
        DataFrame with generated design and metadata in df.attrs

    Raises:
        ValueError: If design type is invalid or design is not feasible
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATE DESIGN FEASIBILITY BEFORE GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    n_vars = len(variables_config)

    # Determine number of levels (for validation)
    n_levels = 2  # Default for most designs
    if design_type == 'full_factorial':
        # Get levels from first variable
        first_var = list(variables_config.keys())[0]
        if 'levels' in variables_config[first_var]:
            n_levels = len(variables_config[first_var]['levels'])
        else:
            n_levels = 3  # Default assumption

    # Validate feasibility
    validation = validate_design_feasibility(design_type, n_vars, n_levels)

    if not validation['feasible']:
        raise ValueError(f"Design generation failed: {validation['message']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CALL APPROPRIATE GENERATOR
    # ═══════════════════════════════════════════════════════════════════════════

    if design_type == 'full_factorial':
        return generate_full_factorial(variables_config)
    elif design_type == 'fractional_factorial_IV':
        return generate_fractional_factorial_IV(variables_config)
    elif design_type == 'fractional_factorial_V':
        return generate_fractional_factorial_V(variables_config)
    elif design_type == 'plackett_burman':
        return generate_plackett_burman(variables_config)
    elif design_type == 'central_composite':
        return generate_central_composite(variables_config)
    else:
        raise ValueError(f"Invalid design_type: {design_type}. "
                        f"Must be one of: 'full_factorial', 'fractional_factorial_IV', 'fractional_factorial_V', "
                        f"'plackett_burman', 'central_composite'")
