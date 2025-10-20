"""
Data export functions for various formats
Handles SAM export and other specialized formats
"""

import pandas as pd


def create_sam_export(data, include_header):
    """
    Create SAM-compatible export format

    Parameters:
    -----------
    data : pd.DataFrame
        Data to export
    include_header : bool
        Whether to include column headers

    Returns:
    --------
    str : SAM-formatted text content
    """
    sam_content = []
    sam_content.append("# SAM Export from CAT Python")
    sam_content.append("# NIR Spectroscopy Data")
    sam_content.append(f"# Samples: {len(data)}")
    sam_content.append(f"# Variables: {len(data.columns)}")
    sam_content.append("#")

    if include_header:
        sam_content.append("# " + "\t".join(data.columns))

    for i, row in data.iterrows():
        row_data = []
        for val in row:
            if pd.isna(val):
                row_data.append("0.0")
            else:
                row_data.append(str(val))
        sam_content.append("\t".join(row_data))

    return '\n'.join(sam_content)
