"""
PCA Interpretation Module

Session management and analysis functions for joint loadings-scores interpretation.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class InterpretationSession:
    """
    Session object for PCA interpretation analysis.

    Stores colors, labels, annotations, and interpretation notes
    for a specific PC pair analysis.
    """

    def __init__(
        self,
        pc_x: str = "PC1",
        pc_y: str = "PC2",
        sample_colors: Optional[Dict[int, str]] = None,
        sample_labels: Optional[Dict[int, str]] = None,
        variable_annotations: Optional[Dict[str, str]] = None,
        interpretation_notes: str = "",
        session_name: str = "default_session"
    ):
        """
        Initialize interpretation session.

        Parameters
        ----------
        pc_x : str
            X-axis principal component (e.g., "PC1")
        pc_y : str
            Y-axis principal component (e.g., "PC2")
        sample_colors : dict, optional
            Sample index to hex color mapping
        sample_labels : dict, optional
            Sample index to label text mapping
        variable_annotations : dict, optional
            Variable name to interpretation notes mapping
        interpretation_notes : str, optional
            General interpretation text
        session_name : str, optional
            Name of this session
        """
        self.pc_x = pc_x
        self.pc_y = pc_y
        self.sample_colors = sample_colors or {}
        self.sample_labels = sample_labels or {}
        self.variable_annotations = variable_annotations or {}
        self.interpretation_notes = interpretation_notes
        self.created_date = datetime.now()
        self.session_name = session_name

    def to_dict(self) -> dict:
        """
        Serialize session to dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary
        """
        return {
            'pc_x': self.pc_x,
            'pc_y': self.pc_y,
            'sample_colors': {str(k): v for k, v in self.sample_colors.items()},
            'sample_labels': {str(k): v for k, v in self.sample_labels.items()},
            'variable_annotations': self.variable_annotations,
            'interpretation_notes': self.interpretation_notes,
            'created_date': self.created_date.isoformat(),
            'session_name': self.session_name
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InterpretationSession':
        """
        Deserialize session from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing session data

        Returns
        -------
        InterpretationSession
            Restored session object
        """
        session = cls(
            pc_x=data.get('pc_x', 'PC1'),
            pc_y=data.get('pc_y', 'PC2'),
            sample_colors={int(k): v for k, v in data.get('sample_colors', {}).items()},
            sample_labels={int(k): v for k, v in data.get('sample_labels', {}).items()},
            variable_annotations=data.get('variable_annotations', {}),
            interpretation_notes=data.get('interpretation_notes', ''),
            session_name=data.get('session_name', 'default_session')
        )

        # Restore date
        if 'created_date' in data:
            try:
                session.created_date = datetime.fromisoformat(data['created_date'])
            except (ValueError, TypeError):
                session.created_date = datetime.now()

        return session

    def save(self, filepath: str) -> bool:
        """
        Save session to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save file

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            filepath_obj = Path(filepath)
            filepath_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath_obj, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    @classmethod
    def load(cls, filepath: str) -> Optional['InterpretationSession']:
        """
        Load session from JSON file.

        Parameters
        ----------
        filepath : str
            Path to load file

        Returns
        -------
        InterpretationSession or None
            Loaded session, or None if failed
        """
        try:
            filepath_obj = Path(filepath)

            if not filepath_obj.exists():
                print(f"File not found: {filepath}")
                return None

            with open(filepath_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return cls.from_dict(data)

        except Exception as e:
            print(f"Error loading session: {e}")
            return None


def get_loading_contributions(
    loadings: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    threshold: float = 0.4
) -> List[Tuple[str, float, float, float]]:
    """
    Get sorted list of variable contributions on PC_x and PC_y.

    Parameters
    ----------
    loadings : pd.DataFrame
        Loading matrix with variables as rows, PCs as columns
    pc_x : str
        X-axis principal component name
    pc_y : str
        Y-axis principal component name
    threshold : float, optional
        Minimum absolute loading to include (default: 0.4)

    Returns
    -------
    list of tuple
        List of (variable, loading_pc_x, loading_pc_y, abs_max)
        sorted by maximum absolute loading descending

    Examples
    --------
    >>> loadings_df = pd.DataFrame({'PC1': [0.8, 0.2], 'PC2': [0.1, 0.9]}, index=['var1', 'var2'])
    >>> get_loading_contributions(loadings_df, 'PC1', 'PC2', threshold=0.3)
    [('var1', 0.8, 0.1, 0.8), ('var2', 0.2, 0.9, 0.9)]
    """
    contributions = []

    for var_name in loadings.index:
        load_x = loadings.loc[var_name, pc_x]
        load_y = loadings.loc[var_name, pc_y]
        abs_max = max(abs(load_x), abs(load_y))

        if abs_max >= threshold:
            contributions.append((var_name, load_x, load_y, abs_max))

    # Sort by maximum absolute loading descending
    contributions.sort(key=lambda x: x[3], reverse=True)

    return contributions


def get_sample_quadrant(
    scores: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    sample_idx: int
) -> Tuple[str, float, float]:
    """
    Determine sample position relative to origin.

    Parameters
    ----------
    scores : pd.DataFrame
        Score matrix with samples as rows, PCs as columns
    pc_x : str
        X-axis principal component name
    pc_y : str
        Y-axis principal component name
    sample_idx : int
        Sample index

    Returns
    -------
    tuple
        (quadrant, score_pc_x, score_pc_y)
        quadrant in ["Q1", "Q2", "Q3", "Q4", "Axis"]

    Examples
    --------
    >>> scores_df = pd.DataFrame({'PC1': [2.0, -1.5], 'PC2': [3.0, 1.0]}, index=[0, 1])
    >>> get_sample_quadrant(scores_df, 'PC1', 'PC2', 0)
    ('Q1', 2.0, 3.0)
    """
    if sample_idx not in scores.index:
        return ("Unknown", 0.0, 0.0)

    score_x = scores.loc[sample_idx, pc_x]
    score_y = scores.loc[sample_idx, pc_y]

    # Determine quadrant
    if abs(score_x) < 0.01 or abs(score_y) < 0.01:
        quadrant = "Axis"
    elif score_x > 0 and score_y > 0:
        quadrant = "Q1"  # Upper-right
    elif score_x < 0 and score_y > 0:
        quadrant = "Q2"  # Upper-left
    elif score_x < 0 and score_y < 0:
        quadrant = "Q3"  # Lower-left
    else:  # score_x > 0 and score_y < 0
        quadrant = "Q4"  # Lower-right

    return (quadrant, score_x, score_y)


def interpret_variable_direction(
    loadings: pd.DataFrame,
    pc_name: str
) -> Dict[str, Dict[str, any]]:
    """
    Get sign and magnitude of each variable on given PC.

    Parameters
    ----------
    loadings : pd.DataFrame
        Loading matrix with variables as rows, PCs as columns
    pc_name : str
        Principal component name (e.g., 'PC1')

    Returns
    -------
    dict
        Dictionary with variable names as keys, each containing:
        - 'sign': '+' or '-'
        - 'magnitude': float (absolute loading)
        - 'description': str (human-readable interpretation)

    Examples
    --------
    >>> loadings_df = pd.DataFrame({'PC1': [0.8, -0.6]}, index=['var1', 'var2'])
    >>> interpret_variable_direction(loadings_df, 'PC1')
    {'var1': {'sign': '+', 'magnitude': 0.8, 'description': 'Positive (0.80)'},
     'var2': {'sign': '-', 'magnitude': 0.6, 'description': 'Negative (-0.60)'}}
    """
    interpretations = {}

    for var_name in loadings.index:
        loading = loadings.loc[var_name, pc_name]
        sign = '+' if loading >= 0 else '-'
        magnitude = abs(loading)

        # Magnitude categories
        if magnitude >= 0.7:
            strength = "Strong"
        elif magnitude >= 0.4:
            strength = "Moderate"
        elif magnitude >= 0.2:
            strength = "Weak"
        else:
            strength = "Negligible"

        description = f"{strength} {sign} ({loading:.2f})"

        interpretations[var_name] = {
            'sign': sign,
            'magnitude': magnitude,
            'description': description
        }

    return interpretations


def joint_sample_variable_interpretation(
    loadings: pd.DataFrame,
    scores: pd.DataFrame,
    sample_idx: int,
    pc_x: str,
    pc_y: str,
    threshold: float = 0.4
) -> str:
    """
    Generate natural language interpretation of sample-variable relationship.

    Describes how a sample relates to important variables based on
    loading directions and score positions.

    Parameters
    ----------
    loadings : pd.DataFrame
        Loading matrix
    scores : pd.DataFrame
        Score matrix
    sample_idx : int
        Sample index to interpret
    pc_x : str
        X-axis principal component
    pc_y : str
        Y-axis principal component
    threshold : float, optional
        Minimum loading magnitude to consider (default: 0.4)

    Returns
    -------
    str
        Natural language interpretation text

    Examples
    --------
    >>> loadings_df = pd.DataFrame({'PC1': [0.8, 0.2], 'PC2': [0.1, 0.9]}, index=['Var1', 'Var2'])
    >>> scores_df = pd.DataFrame({'PC1': [2.3], 'PC2': [1.5]}, index=[0])
    >>> joint_sample_variable_interpretation(loadings_df, scores_df, 0, 'PC1', 'PC2')
    'Sample 0 is positioned HIGH on PC1 (score=2.30) and HIGH on PC2 (score=1.50)...'
    """
    if sample_idx not in scores.index:
        return f"Sample {sample_idx} not found in scores."

    # Get sample position
    quadrant, score_x, score_y = get_sample_quadrant(scores, pc_x, pc_y, sample_idx)

    # Get important variables
    contributions = get_loading_contributions(loadings, pc_x, pc_y, threshold)

    if not contributions:
        return f"Sample {sample_idx} has no strong variable associations (threshold={threshold})."

    # Build interpretation text
    text_parts = []

    # Sample position
    pos_x = "HIGH" if score_x > 0 else "LOW"
    pos_y = "HIGH" if score_y > 0 else "LOW"

    text_parts.append(
        f"Sample {sample_idx} is positioned {pos_x} on {pc_x} (score={score_x:.2f}) "
        f"and {pos_y} on {pc_y} (score={score_y:.2f})."
    )

    # Variables on PC_x
    vars_pcx_pos = [v for v, lx, ly, _ in contributions if lx > threshold]
    vars_pcx_neg = [v for v, lx, ly, _ in contributions if lx < -threshold]

    if vars_pcx_pos:
        text_parts.append(
            f"\nOn {pc_x}, this sample has HIGH levels of: {', '.join(vars_pcx_pos)}."
        )

    if vars_pcx_neg:
        text_parts.append(
            f"\nOn {pc_x}, this sample has LOW levels of: {', '.join(vars_pcx_neg)}."
        )

    # Variables on PC_y
    vars_pcy_pos = [v for v, lx, ly, _ in contributions if ly > threshold]
    vars_pcy_neg = [v for v, lx, ly, _ in contributions if ly < -threshold]

    if vars_pcy_pos:
        text_parts.append(
            f"\nOn {pc_y}, this sample has HIGH levels of: {', '.join(vars_pcy_pos)}."
        )

    if vars_pcy_neg:
        text_parts.append(
            f"\nOn {pc_y}, this sample has LOW levels of: {', '.join(vars_pcy_neg)}."
        )

    return ''.join(text_parts)
