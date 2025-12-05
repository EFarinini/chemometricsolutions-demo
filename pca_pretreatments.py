"""
PCA Pretreatment Detection Module - SIMPLIFIED

This module provides functionality to:
1. Detect which pretreatments have been applied to a dataset (from transformation_history)
2. Display pretreatment information to the user
3. Warn users to apply the same pretreatments to test data

Note: PCA's own centering/scaling already uses training statistics correctly.
This module is INFORMATIONAL ONLY - it does not automatically apply pretreatments.

Author: ChemometricSolutions
"""

import streamlit as st
from typing import Dict, Any, Optional


class PretreatmentInfo:
    """
    Simple class to detect and display pretreatment information.

    This is INFORMATIONAL ONLY - does not save statistics or apply transformations.
    """

    def __init__(self):
        self.pretreatments = []  # List of detected pretreatment steps
        self.dataset_name = None
        self.original_dataset_name = None

    def detect_pretreatments(self, dataset_name: str, transformation_history: Dict) -> bool:
        """
        Detect which pretreatments have been applied to a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the current dataset
        transformation_history : dict
            Session state transformation_history dictionary

        Returns
        -------
        bool
            True if pretreatments were detected, False otherwise
        """
        self.dataset_name = dataset_name
        self.pretreatments = []

        # Check if this dataset is in transformation history
        if dataset_name not in transformation_history:
            return False

        # Get transformation info
        transform_info = transformation_history[dataset_name]

        pretreatment = {
            'name': transform_info.get('transform', 'Unknown'),
            'transform_type': transform_info.get('transform_type', 'unknown'),
            'params': transform_info.get('params', {}),
            'col_range': transform_info.get('col_range', None),
            'original_dataset': transform_info.get('original_dataset', None),
            'timestamp': transform_info.get('timestamp', None)
        }

        self.pretreatments.append(pretreatment)
        self.original_dataset_name = transform_info.get('original_dataset', None)

        return True

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of detected pretreatments.

        Returns
        -------
        dict
            Summary information about pretreatments
        """
        summary = {
            'dataset_name': self.dataset_name,
            'original_dataset': self.original_dataset_name,
            'n_pretreatments': len(self.pretreatments),
            'pretreatments': []
        }

        for pretreatment in self.pretreatments:
            summary['pretreatments'].append({
                'name': pretreatment['name'],
                'type': pretreatment['transform_type'],
                'params': pretreatment['params']
            })

        return summary


# Helper functions

def detect_pretreatments(dataset_name: str, transformation_history: Optional[Dict] = None) -> Optional[PretreatmentInfo]:
    """
    Convenience function to detect pretreatments.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    transformation_history : dict, optional
        Transformation history from session state

    Returns
    -------
    PretreatmentInfo or None
        Info with detected pretreatments, or None if no pretreatments found
    """
    if transformation_history is None:
        if 'transformation_history' in st.session_state:
            transformation_history = st.session_state.transformation_history
        else:
            return None

    pretreat_info = PretreatmentInfo()

    if pretreat_info.detect_pretreatments(dataset_name, transformation_history):
        return pretreat_info
    else:
        return None


def display_pretreatment_info(pretreat_info: PretreatmentInfo, context: str = "training"):
    """
    Display pretreatment information in Streamlit UI.

    Parameters
    ----------
    pretreat_info : PretreatmentInfo
        The info with pretreatment information
    context : str
        Either "training" or "testing" to customize the message
    """
    if not pretreat_info.pretreatments:
        st.info("📊 No pretreatments detected - using raw data")
        return

    st.success(f"✅ **Pretreatments detected**: {len(pretreat_info.pretreatments)} transformation(s)")

    for i, pretreatment in enumerate(pretreat_info.pretreatments, 1):
        with st.expander(f"Pretreatment {i}: {pretreatment['name']}", expanded=True):
            st.write(f"**Type**: {pretreatment['transform_type']}")

            if pretreatment['params']:
                st.write("**Parameters**:")
                for param_name, param_value in pretreatment['params'].items():
                    st.write(f"  • {param_name}: {param_value}")

            if pretreatment['col_range']:
                col_start = pretreatment['col_range'][0] + 1
                col_end = pretreatment['col_range'][1]
                st.write(f"**Column range**: {col_start} to {col_end}")

            if pretreatment['timestamp']:
                st.write(f"**Applied**: {pretreatment['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Context-specific messages
    if context == "training":
        st.info("💡 **Note**: PCA's centering/scaling will use training statistics automatically")
        st.warning("⚠️ **Important**: Test data must have the same pretreatment applied!")

    elif context == "testing":
        st.error("🚨 **ACTION REQUIRED**: Apply the same pretreatment to your test data!")
        st.info("💡 Use the Transformations page to apply the same transformation with identical parameters")


def display_pretreatment_warning(training_pretreat: PretreatmentInfo, test_dataset_name: str):
    """
    Display warning about pretreatment requirements for test data.

    Parameters
    ----------
    training_pretreat : PretreatmentInfo
        Pretreatments that were applied to training data
    test_dataset_name : str
        Name of the test dataset being used
    """
    if not training_pretreat or not training_pretreat.pretreatments:
        return

    st.markdown("### 🔬 Pretreatment Check")

    st.warning("⚠️ **Training data was pretreated** - verify test data has the same pretreatment!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training pretreatments:**")
        for pretreatment in training_pretreat.pretreatments:
            st.write(f"✓ {pretreatment['name']}")
            if pretreatment['params']:
                for param_name, param_value in pretreatment['params'].items():
                    st.caption(f"  {param_name}: {param_value}")

    with col2:
        st.markdown("**Test dataset:**")
        st.write(f"📊 {test_dataset_name}")

        # Check if test dataset is also in transformation history
        test_pretreat = detect_pretreatments(
            test_dataset_name,
            st.session_state.get('transformation_history', {})
        )

        if test_pretreat and test_pretreat.pretreatments:
            st.success("✅ Pretreatments detected on test data:")
            for pretreatment in test_pretreat.pretreatments:
                st.write(f"✓ {pretreatment['name']}")
        else:
            st.error("❌ No pretreatments detected on test data!")
            st.warning("Apply the same pretreatment in Transformations page first")

    # Helpful instructions
    with st.expander("📖 How to apply pretreatments to test data", expanded=False):
        st.markdown("""
        **Steps to preprocess test data:**

        1. Go to **Data Handling** page
        2. Load your test dataset
        3. Go to **Transformations** page
        4. Apply the **same transformation** as training data
        5. Use the **same parameters** (window size, etc.)
        6. Use the **same column range**
        7. Save the transformed test dataset
        8. Return to **Quality Control** > **Testing & Monitoring**
        9. Select the transformed test dataset

        **Important**:
        - Use identical transformation parameters
        - PCA's centering/scaling will automatically use training statistics
        - Only the spectral pretreatments (SNV, derivatives, etc.) need manual application
        """)
