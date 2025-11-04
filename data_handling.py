"""
CAT Data Handling Page - UPDATED VERSION WITH TRANSFORMATION MEMORY
Equivalent to DH_* R scripts for data import/export and workspace management
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import openpyxl for Excel export
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

def safe_join(items, separator="\n"):
    """Safely join list items as strings, handling any data type"""
    if not items:
        return ""
    try:
        # Convert all items to strings, handle None values
        safe_items = []
        for item in items:
            if item is None:
                safe_items.append("")
            elif isinstance(item, (list, tuple)):
                safe_items.append(str(item))
            elif isinstance(item, dict):
                safe_items.append(str(item))
            else:
                safe_items.append(str(item))
        return separator.join(safe_items)
    except Exception as e:
        # Fallback: convert everything to string
        return separator.join([str(x) if x is not None else "" for x in items])

def safe_format_objects(objects_list):
    """Safely format objects list for display"""
    if not objects_list:
        return []
    
    formatted = []
    for obj in objects_list:
        try:
            formatted.append(str(obj))
        except:
            formatted.append("Object (display error)")
    return formatted

# Helper functions for different file formats
def _load_csv_txt(uploaded_file, separator, decimal, encoding, has_header, has_index, skip_rows, skip_cols, na_values, quote_char):
    """Load CSV/TXT files with robust encoding detection"""
    encodings = [encoding, 'latin-1', 'cp1252', 'utf-8', 'ascii']
    data = None
    na_list = [x.strip() for x in na_values.split(',') if x.strip()]
    quote_setting = None if quote_char == "None" else quote_char

    for enc in encodings:
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)

            data = pd.read_csv(uploaded_file, 
                             sep=separator,
                             header=0 if has_header else None,
                             index_col=0 if has_index else None,
                             encoding=enc,
                             skiprows=skip_rows if skip_rows > 0 else None,
                             usecols=lambda x: x >= skip_cols if isinstance(x, int) else True,
                             na_values=na_list,
                             decimal=decimal,
                             quotechar=quote_setting)
            st.info(f"File loaded with encoding: {enc}")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    if data is None:
        raise ValueError("Unable to decode file with any encoding")
    
    # CORREZIONE: Se non c'Ã¨ una colonna indice, usa indici 1-based
    if not has_index:
        data.index = range(1, len(data) + 1)
        data.index.name = None  # Rimuovi il nome dell'indice
    
    return data

def _load_spectral_data(uploaded_file, separator, decimal, encoding, has_header, has_index, skip_rows, data_format, wavelength_info):
    """Load spectral data (DAT/ASC files)"""
    data = _load_csv_txt(uploaded_file, separator, decimal, encoding, has_header, has_index, skip_rows, 0, "NA", '"')
    
    if data_format == "Transposed (variablesÃ—samples)":
        data = data.T
        
        # Rename columns to be more informative
        if len(data.columns) > 100:  # Likely spectral data
            # Create wavelength-like column names
            wavelengths = np.linspace(400, 4000, len(data.columns))
            new_columns = [f"WL_{wl:.1f}" for wl in wavelengths]
            data.columns = new_columns
        else:
            # Generic variable names
            data.columns = [f"Var_{i+1}" for i in range(len(data.columns))]
        
        st.success("Data transposed: variablesÃ—samples â†' samplesÃ—variables")
        st.info(f"Final format: {data.shape[0]} samples Ã— {data.shape[1]} variables")
        
        # CORREZIONE: Dopo la trasposizione, forza indici 1-based se necessario
        if not has_index:
            data.index = range(1, len(data) + 1)
            data.index.name = None
    
    if wavelength_info:
        st.info("Wavelength/frequency information detected in headers")
    
    return data

def _load_raw_data(uploaded_file, encoding="utf-8"):
    """Load RAW files (XRD spectra) - Enhanced version"""
    try:
        # Read content
        content = uploaded_file.read()
        file_name = uploaded_file.name.split('.')[0]
        
        # Try different approaches
        st.info(f"Processing RAW file: {uploaded_file.name}")
        st.info(f"File size: {len(content)} bytes")
        
        # Method 1: Try as text first
        try:
            if isinstance(content, bytes):
                # Try multiple encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
                content_str = None
                
                for enc in encodings:
                    try:
                        content_str = content.decode(enc, errors='ignore')
                        st.info(f"Decoded with {enc} encoding")
                        break
                    except:
                        continue
                
                if content_str is None:
                    raise ValueError("Could not decode file with any encoding")
            else:
                content_str = content
            
            # Split into lines
            lines = content_str.split('\n')
            st.info(f"Found {len(lines)} lines in file")
            
            # Look for numerical data
            data_lines = []
            metadata_lines = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check if line contains numerical data
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        # Try to convert to numbers
                        numbers = []
                        for part in parts:
                            # Remove common non-numeric characters
                            clean_part = part.replace(',', '.').replace(';', '')
                            numbers.append(float(clean_part))
                        
                        if len(numbers) >= 1:
                            data_lines.append(numbers)
                    except (ValueError, TypeError):
                        # Not numerical data, treat as metadata
                        metadata_lines.append(line)
            
            st.info(f"Found {len(data_lines)} data lines")
            st.info(f"Found {len(metadata_lines)} metadata lines")
            
            if len(data_lines) > 0:
                # Determine data structure
                max_cols = max(len(row) for row in data_lines)
                
                # Pad all rows to same length
                padded_data = []
                for row in data_lines:
                    padded_row = row + [np.nan] * (max_cols - len(row))
                    padded_data.append(padded_row)
                
                # Create appropriate column names
                if max_cols == 1:
                    # Single column - intensity data
                    columns = ['Intensity']
                    # Add angle column
                    angles = np.linspace(5, 80, len(padded_data))
                    padded_data = [[angles[i]] + row for i, row in enumerate(padded_data)]
                    columns = ['2Theta'] + columns
                elif max_cols == 2:
                    # Two columns - angle and intensity
                    columns = ['2Theta', 'Intensity']
                elif max_cols == 3:
                    # Three columns - angle, intensity, background
                    columns = ['2Theta', 'Intensity', 'Background']
                else:
                    # Multiple columns
                    columns = ['2Theta', 'Intensity'] + [f'Col_{i}' for i in range(2, max_cols)]
                
                # Create DataFrame
                data = pd.DataFrame(padded_data, columns=columns)
                
                # Add sample info
                data.insert(0, 'Sample_ID', file_name)
                
                st.success(f"RAW file parsed successfully!")
                st.info(f"Final shape: {data.shape[0]} points Ã— {data.shape[1]} variables")
                
                return data
            
            else:
                st.warning("No numerical data found in text parsing")
                
        except Exception as e:
            st.warning(f"Text parsing failed: {str(e)}")
        
        # Method 2: Try as binary data
        try:
            st.info("Attempting binary parsing...")
            
            # For binary files, try to extract numerical data
            if len(content) > 100:
                import struct
                
                # Try different binary formats
                for fmt in ['<f', '<d', '<i', '<h', '>f', '>d', '>i', '>h']:
                    try:
                        step = struct.calcsize(fmt)
                        n_values = len(content) // step
                        
                        if n_values > 10:  # At least 10 data points
                            values = struct.unpack(f'{fmt[0]}{n_values}{fmt[-1]}', content[:n_values*step])
                            
                            # Filter reasonable values
                            valid_values = [v for v in values if -1e6 < v < 1e6 and not np.isnan(v)]
                            
                            if len(valid_values) > 10:
                                st.info(f"Binary format {fmt}: found {len(valid_values)} valid values")
                                
                                # Create appropriate structure
                                if len(valid_values) > 100:
                                    # Assume it's XRD data - CREATE CHEMOMETRIC FORMAT
                                    angles = np.linspace(5, 80, len(valid_values))
                                    
                                    # Create chemometric format: samples Ã— variables
                                    # One row = one sample, columns = 2Theta angles with intensities
                                    data_dict = {'Sample_ID': file_name}
                                    
                                    for i, (angle, intensity) in enumerate(zip(angles, valid_values)):
                                        data_dict[f'2Theta_{angle:.2f}'] = intensity
                                    
                                    # Create DataFrame with single row (one sample)
                                    data = pd.DataFrame([data_dict])
                                    
                                    st.success(f"XRD spectrum loaded in chemometric format!")
                                    st.info(f"2Î¸ range: {angles[0]:.2f}Â° to {angles[-1]:.2f}Â°")
                                    st.info(f"Format: 1 sample Ã— {len(valid_values)} variables (2Î¸ angles)")
                                else:
                                    # Short data - create chemometric format
                                    data_dict = {'Sample_ID': file_name}
                                    
                                    for i, val in enumerate(valid_values):
                                        data_dict[f'Point_{i+1}'] = val
                                    
                                    # Create DataFrame with single row
                                    data = pd.DataFrame([data_dict])
                                
                                st.success(f"Binary RAW data extracted!")
                                st.info(f"Shape: {data.shape[0]} points Ã— {data.shape[1]} variables")
                                
                                return data
                                
                    except Exception:
                        continue
                        
        except Exception as e:
            st.warning(f"Binary parsing failed: {str(e)}")
        
        # Method 3: Create minimal fallback
        st.warning("Could not parse RAW file - creating minimal dataset")
        
        # Create a minimal dataset with file info
        fallback_data = pd.DataFrame({
            'Sample_ID': [file_name],
            'File_Name': [uploaded_file.name],
            'File_Size_bytes': [len(content)],
            'Status': ['Parsing_Failed'],
            'Suggestion': ['Try converting to TXT or CSV format first']
        })
        
        st.info("**Suggestions:**")
        st.info("• Try exporting your RAW file as TXT or CSV from the original software")
        st.info("• Check if the file is corrupted")
        st.info("• Contact support with file format details")
        
        return fallback_data
        
    except Exception as e:
        st.error(f"RAW file processing failed: {str(e)}")
        
        # Final fallback
        fallback_data = pd.DataFrame({
            'Sample_ID': [uploaded_file.name.split('.')[0]],
            'Status': ['Critical_Error'],
            'Error': [str(e)],
            'Suggestion': ['Contact support or try different format']
        })
        
        return fallback_data

def _load_excel_data(uploaded_file, sheet_name, skip_rows, skip_cols, has_header, has_index, na_values_excel):
    """Load Excel files with enhanced parameters"""
    try:
        sheet_num = int(sheet_name)
    except ValueError:
        sheet_num = sheet_name

    na_list_excel = [x.strip() for x in na_values_excel.split(',') if x.strip()]

    # Reset file pointer to beginning
    uploaded_file.seek(0)

    data = pd.read_excel(uploaded_file,
                       sheet_name=sheet_num,
                       header=0 if has_header else None,
                       index_col=0 if has_index else None,
                       skiprows=skip_rows,
                       na_values=na_list_excel)
    
    # Handle skip_cols for Excel after loading
    if skip_cols > 0:
        data = data.iloc[:, skip_cols:]
    
    # CORREZIONE: Se non c'Ã¨ una colonna indice, usa indici 1-based
    if not has_index:
        data.index = range(1, len(data) + 1)
        data.index.name = None
    
    return data

def _save_original_to_history(data, dataset_name):
    """Save original dataset to transformation history for reference"""
    if 'transformation_history' not in st.session_state:
        st.session_state.transformation_history = {}
    
    # Save original only if not exists
    original_name = f"{dataset_name.split('.')[0]}_ORIGINAL"
    
    if original_name not in st.session_state.transformation_history:
        st.session_state.transformation_history[original_name] = {
            'data': data.copy(),
            'transform': 'Original (Untransformed)',
            'params': {},
            'col_range': None,
            'timestamp': pd.Timestamp.now(),
            'transform_type': 'original'
        }

# Export helper functions
def _create_sam_export(data, include_header):
    """Create SAM-compatible export format"""
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

def _parse_clipboard_data(clipboard_text, separator, decimal, has_header, has_index, na_values):
    """Parse clipboard data into pandas DataFrame"""
    try:
        # Split into lines
        lines = clipboard_text.strip().split('\n')
        
        if not lines:
            raise ValueError("No data found in clipboard")
        
        # Process NA values
        na_list = [x.strip() for x in na_values.split(',') if x.strip()]
        
        # Detect separator if auto-detection is enabled
        if separator == "Auto-detect":
            # Smart separator detection that prioritizes structural delimiters
            # NEVER auto-select space (internal spaces in column names are common)
            first_line = lines[0]

            # Priority order: Tab → Comma → Semicolon (NEVER space for auto-detect)
            separators_priority = ['\t', ',', ';']
            separator_counts = {sep: first_line.count(sep) for sep in separators_priority}

            # Find separator with highest count (minimum 1)
            valid_separators = {sep: count for sep, count in separator_counts.items() if count > 0}

            if valid_separators:
                # Select separator with highest count
                separator = max(valid_separators, key=valid_separators.get)
            else:
                # No standard separators found, fallback to comma
                separator = ','
        
        # Parse data manually
        rows = []
        for line in lines:
            # Split by separator
            if separator == ' ':
                # Handle multiple spaces
                row = [cell.strip() for cell in line.split() if cell.strip()]
            else:
                row = [cell.strip() for cell in line.split(separator)]
            rows.append(row)
        
        # Find maximum row length for padding
        max_cols = max(len(row) for row in rows) if rows else 0
        
        # Pad rows to same length
        for row in rows:
            while len(row) < max_cols:
                row.append('')
        
        # Convert to DataFrame
        if has_header and len(rows) > 0:
            columns = rows[0]
            data_rows = rows[1:]
        else:
            columns = [f"Col_{i+1}" for i in range(max_cols)]
            data_rows = rows

        # Normalize column names: strip whitespace, remove special chars
        columns_normalized = []
        for col in columns:
            # Strip leading/trailing whitespace
            col_clean = str(col).strip()
            # Replace multiple spaces/newlines/tabs with underscore
            col_clean = col_clean.replace('\n', '_').replace('\t', '_').replace('\r', '_')
            col_clean = '_'.join(col_clean.split())  # Multiple spaces → single underscore
            # Remove problematic characters but keep numbers and letters
            col_clean = ''.join(c if c.isalnum() or c in '_.' else '_' for c in col_clean)
            # Remove leading numbers and double underscores
            while col_clean.startswith('_'):
                col_clean = col_clean[1:]
            while '__' in col_clean:
                col_clean = col_clean.replace('__', '_')
            # Ensure non-empty
            col_clean = col_clean if col_clean else f"Col_{columns.index(col)+1}"
            columns_normalized.append(col_clean)

        # Handle duplicate column names
        seen = {}
        final_columns = []
        for col in columns_normalized:
            if col in seen:
                seen[col] += 1
                final_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_columns.append(col)

        # Create DataFrame with cleaned columns
        if not data_rows:
            data = pd.DataFrame(columns=final_columns)
        else:
            data = pd.DataFrame(data_rows, columns=final_columns)
        
        # Handle index column
        if has_index and len(data.columns) > 0:
            data = data.set_index(data.columns[0])
        elif not has_index:
            # Use 1-based index
            data.index = range(1, len(data) + 1)
            data.index.name = None
        
        # Convert numeric columns
        for col in data.columns:
            # Try to convert to numeric, replacing NA values
            numeric_data = pd.to_numeric(data[col], errors='coerce')
            
            # If more than 50% of values are numeric, treat as numeric column
            if numeric_data.notna().sum() / len(data) > 0.5:
                data[col] = numeric_data
            
            # Replace NA values
            for na_val in na_list:
                data[col] = data[col].replace(na_val, np.nan)
        
        # Handle decimal separator conversion
        if decimal == ',':
            for col in data.select_dtypes(include=['object']).columns:
                try:
                    # Try to convert comma decimals to dot decimals
                    data[col] = data[col].astype(str).str.replace(',', '.', regex=False)
                    data[col] = pd.to_numeric(data[col], errors='ignore')
                except:
                    pass
        
        return data, separator
        
    except Exception as e:
        raise ValueError(f"Error parsing clipboard data: {str(e)}")

def show():
    """Display the Data Handling page"""
    
    st.markdown("# Data Handling")
    st.markdown("*Import, export, and manage your datasets*")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Load Data", 
        "Export Data", 
        "Workspace", 
        "Dataset Operations", 
        "Randomize",
        "Metadata Management"
    ])
    
    # ===== LOAD DATA TAB =====
    with tab1:
        st.markdown("## Load Data")
        st.markdown("*Equivalent to DH_load_* R scripts*")
        
        load_method = st.selectbox(
            "Choose loading method:",
            ["Upload File", "Multi-File Upload & Merge", "Copy/Paste", "URL", "Sample Data", "Format Info"],
            key="load_method_select"
        )
        
        if load_method == "Format Info":
            st.markdown("### Supported File Formats in Chemical Analysis")
            
            with st.expander("**Standard Data Formats**"):
                st.markdown("""
                - **CSV**: Comma-separated values (universal)
                - **TXT**: Tab-delimited text files
                - **Excel (XLS/XLSX)**: Microsoft Excel spreadsheets
                - **JSON**: JavaScript Object Notation
                """)
            
            with st.expander("**Spectroscopy & Analytical Chemistry**"):
                st.markdown("""
                - **DAT**: Spectral/instrumental data files
                - **ASC**: ASCII spectroscopy data
                - **SPC**: Galactic SPC format (binary spectroscopy)
                - **SAM**: NIR spectra (MNIR format) - **PERFECT FOR CONVERSION TO XLSX!**
                - **RAW**: XRD diffraction data - **NEW! X-ray diffraction spectra**
                - **JDX/DX**: JCAMP-DX standard format
                - **PRN**: Formatted text (space-delimited)
                """)
            
            with st.expander("**Specialized Formats**"):
                st.markdown("""
                - **ARFF**: Weka machine learning format
                - **TSV**: Tab-separated values
                - **ODS**: OpenDocument spreadsheet
                - **H5/HDF5**: Hierarchical data format (large datasets)
                - **MAT**: MATLAB data files
                """)
            
            st.success("**Perfect for converting instrumental files to Excel format!**")
            
            # Professional Services Note
            st.markdown("---")
            st.info("""
            💡 **Need support for additional formats or custom import/export workflows?**  
            Professional data handling solutions available at [chemometricsolutions.com](https://chemometricsolutions.com)
            
            🔧 **Enterprise features:** Custom parsers • Batch processing • Database integration • API connectivity
            """)
        
        elif load_method == "Upload File":
            st.markdown("### File Upload")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'txt', 'xls', 'xlsx', 'json', 'dat', 'asc', 'spc', 'raw',
                     'prn', 'tsv', 'jdx', 'dx', 'arff', 'ods', 'h5', 'hdf5', 'mat']
            )
            
            if uploaded_file is not None:
                # Auto-detect format
                file_ext = uploaded_file.name.lower().split('.')[-1]
                
                format_map = {
                    'csv': 'CSV',
                    'txt': 'TXT (Tab-delimited)',
                    'xls': 'Excel (XLS/XLSX)',
                    'xlsx': 'Excel (XLS/XLSX)',
                    'json': 'JSON',
                    'dat': 'DAT (Spectral/Instrumental Data)',
                    'asc': 'ASC (ASCII Data)',
                    'spc': 'SPC (Spectroscopy)',
                    'raw': 'RAW (XRD Diffraction)',
                    'prn': 'PRN (Formatted Text)',
                    'tsv': 'TSV (Tab-separated)',
                    'jdx': 'JDX/DX (JCAMP-DX)',
                    'dx': 'JDX/DX (JCAMP-DX)',
                    'arff': 'ARFF (Weka Format)',
                    'ods': 'ODS (OpenDocument)',
                    'h5': 'H5/HDF5 (Hierarchical Data)',
                    'hdf5': 'H5/HDF5 (Hierarchical Data)',
                    'mat': 'MAT (MATLAB Format)'
                }
                
                file_format = format_map.get(file_ext, 'CSV')
                st.success(f"**Auto-detected format**: {file_format}")
                
                # Override option
                if st.checkbox("Override format detection"):
                    file_format = st.selectbox("Select different format:", list(format_map.values()))
                
                # Basic parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    has_header = st.checkbox("First row contains headers", value=True)
                    has_index = st.checkbox("First column contains row names", value=False)
                
                with col2:
                    if file_format in ["CSV", "TXT (Tab-delimited)"]:
                        separator = st.selectbox("Separator:", [",", ";", "\t", " "], key="sep_basic")
                        decimal = st.selectbox("Decimal separator:", [".", ","], key="dec_basic")
                        encoding = st.selectbox("Encoding:", ["utf-8", "latin-1", "cp1252"], key="enc_basic")
                    elif file_format == "Excel (XLS/XLSX)":
                        # Excel-specific options: Sheet selection
                        try:
                            # Get list of sheet names
                            excel_file = pd.ExcelFile(uploaded_file)
                            sheet_names = excel_file.sheet_names
                            st.success(f"📊 **{len(sheet_names)} sheet(s) detected**: {', '.join(sheet_names)}")

                            # Sheet selector
                            selected_sheet = st.selectbox(
                                "📑 Select sheet to load:",
                                sheet_names,
                                key="excel_sheet_selector"
                            )

                            # Preview selected sheet
                            if selected_sheet:
                                st.markdown("### 👀 Sheet Preview")
                                try:
                                    preview_data = pd.read_excel(uploaded_file, sheet_name=selected_sheet, nrows=5)

                                    # Display info stacked vertically
                                    numeric_cols = preview_data.select_dtypes(include=[np.number]).columns
                                    st.info(f"**Shape**: {preview_data.shape[0]} rows × {preview_data.shape[1]} columns (preview only) | **Numeric columns**: {len(numeric_cols)}")

                                    st.dataframe(preview_data.head(5), height=200, use_container_width=True)

                                    with st.expander("📋 Data Types"):
                                        dtypes_df = pd.DataFrame({
                                            'Column': preview_data.columns,
                                            'Type': preview_data.dtypes.astype(str)
                                        })
                                        st.dataframe(dtypes_df, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not preview sheet: {str(e)}")

                        except Exception as e:
                            st.warning(f"Could not read sheet names: {str(e)}")
                            selected_sheet = "0"  # Fallback to first sheet

                        separator = ","
                        decimal = "."
                        encoding = "utf-8"
                    else:
                        separator = ","
                        decimal = "."
                        encoding = "utf-8"
                        selected_sheet = "0"

                # Advanced options
                with st.expander("Advanced Options"):
                    skip_rows = st.number_input("Skip top rows:", min_value=0, value=0)
                    skip_cols = st.number_input("Skip left columns:", min_value=0, value=0)
                    na_values = st.text_input("Missing value indicators:", value="NA")
                
                # Load button
                if st.button("Load Data"):
                    try:
                        # Load based on format
                        if file_format == "CSV":
                            data = _load_csv_txt(uploaded_file, separator, decimal, encoding,
                                                has_header, has_index, skip_rows, skip_cols, na_values, '"')
                        elif file_format == "TXT (Tab-delimited)":
                            data = _load_csv_txt(uploaded_file, '\t', decimal, encoding,
                                                has_header, has_index, skip_rows, skip_cols, na_values, '"')
                        elif file_format == "Excel (XLS/XLSX)":
                            # ===== LOAD EXCEL FILE WITH SHEET SELECTION =====
                            # Use selected sheet (or "0" as fallback)
                            sheet_to_load = selected_sheet if 'selected_sheet' in locals() else "0"
                            data = _load_excel_data(uploaded_file, sheet_to_load, skip_rows, skip_cols,
                                                  has_header, has_index, na_values)

                            # FORMAT NAME: filename_sheetname (e.g., "washing_train")
                            base_name = Path(uploaded_file.name).stem  # Remove .xlsx
                            formatted_name = f"{base_name}_{sheet_to_load}".lower()

                            # SAVE formatted name in session state
                            st.session_state.current_data = data
                            st.session_state.current_dataset = formatted_name
                            st.session_state.dataset_name = formatted_name

                            # Save original to transformation history
                            _save_original_to_history(data, formatted_name)

                            st.success(f"✅ Loaded sheet: **{sheet_to_load}** → Dataset name: **{formatted_name}**")
                        elif file_format == "JSON":
                            data = pd.read_json(uploaded_file)
                        elif file_format == "DAT (Spectral/Instrumental Data)":
                            data = _load_spectral_data(uploaded_file, '\t', decimal, encoding,
                                                     has_header, has_index, skip_rows, "Matrix (samples×variables)", False)
                        elif file_format == "ASC (ASCII Data)":
                            data = _load_spectral_data(uploaded_file, '\t', decimal, encoding,
                                                     has_header, has_index, skip_rows, "Matrix (samples×variables)", False)
                        elif file_format == "RAW (XRD Diffraction)":
                            data = _load_raw_data(uploaded_file, encoding)
                        else:
                            data = _load_csv_txt(uploaded_file, separator, decimal, encoding,
                                                has_header, has_index, skip_rows, skip_cols, na_values, '"')

                        # === VALIDATE LOADED DATA ===
                        if len(data) == 0:
                            st.error("❌ Dataset is empty - no samples available!")
                            st.info("📊 The loaded file contains no data rows")
                            st.info("📊 Action: Check the file format or load a different file")
                            return

                        # Store data (skip for Excel as it's already handled above)
                        if file_format != "Excel (XLS/XLSX)":
                            st.session_state.current_data = data
                            st.session_state.current_dataset = uploaded_file.name
                            # Save original to transformation history
                            _save_original_to_history(data, uploaded_file.name)

                        st.success(f"✅ Data loaded successfully: {data.shape[0]} rows × {data.shape[1]} columns")

                        # Data Overview Section
                        st.markdown("### 📊 Data Overview")

                        # Key metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Samples", data.shape[0])
                        with col2:
                            st.metric("Variables", data.shape[1])
                        with col3:
                            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                            st.metric("Numeric", len(numeric_cols))
                        with col4:
                            st.metric("Missing", data.isnull().sum().sum())
                        with col5:
                            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                            st.metric("Size", f"{memory_mb:.1f} MB")

                        # Data types summary
                        with st.expander("📋 **Column Types Summary**"):
                            type_counts = data.dtypes.value_counts()
                            col_type1, col_type2 = st.columns(2)

                            with col_type1:
                                st.markdown("**Data Types:**")
                                for dtype, count in type_counts.items():
                                    st.write(f"• {dtype}: {count} columns")

                            with col_type2:
                                st.markdown("**Variable Categories:**")
                                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                                st.write(f"• Numeric: {len(numeric_cols)}")
                                st.write(f"• Categorical: {len(categorical_cols)}")
                                st.write(f"• Total: {len(data.columns)}")

                        # Data preview
                        st.markdown("### 📄 Data Preview")
                        st.dataframe(data.head(10), use_container_width=True)

                        # Additional statistics
                        if len(numeric_cols) > 0:
                            with st.expander("📈 **Numeric Variables Statistics**"):
                                stats_df = data[numeric_cols].describe().T
                                st.dataframe(stats_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
        
        elif load_method == "Copy/Paste":
            st.markdown("### Copy/Paste Data")
            st.markdown("*Copy data from Excel, web tables, or any text source and paste directly*")
            
            # Instructions
            with st.expander("📋 **How to use Copy/Paste**"):
                st.markdown("""
                **Instructions:**
                1. **Copy data** from Excel, Google Sheets, web tables, or any text source
                2. **Paste below** using Ctrl+V (Windows) or Cmd+V (Mac) 
                3. **Configure settings** if needed
                4. **Load data** into CAT Python
                
                **Supported Sources:**
                - Excel spreadsheets (copy selected cells)
                - Google Sheets / LibreOffice Calc
                - Web tables from research papers
                - Text files with delimited data
                - Scientific instrument output (copy from software)
                - Spectral data from equipment software
                
                **Tips:**
                - Data will be auto-formatted for chemometric analysis
                - Column headers and row names are automatically detected
                - Mixed data types (numbers + text) are handled correctly
                - Missing values (empty cells) are converted to NaN
                """)
            
            # Text area for pasting
            clipboard_data = st.text_area(
                "📋 **Paste your data here**:",
                height=200,
                placeholder="Paste your data here using Ctrl+V or Cmd+V...\n\nExample:\nSample_ID\tWave_1000\tWave_1100\tWave_1200\nSample1\t0.534\t0.623\t0.445\nSample2\t0.612\t0.534\t0.523\nSample3\t0.445\t0.678\t0.599",
                key="clipboard_input"
            )
            
            if clipboard_data.strip():
                newline_char = '\n'
                st.success(f"✅ **Data detected**: {len(clipboard_data.strip().split(newline_char))} lines")
                
                # Configuration options
                st.markdown("#### ⚙️ Parse Settings")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    separator_paste = st.selectbox(
                        "Separator:", 
                        ["Auto-detect", "\t (Tab)", ", (Comma)", "; (Semicolon)", "  (Space)"],
                        key="sep_paste"
                    )
                    
                    # Convert display names to actual separators
                    sep_map = {
                        "Auto-detect": "Auto-detect",
                        "\t (Tab)": "\t",
                        ", (Comma)": ",",
                        "; (Semicolon)": ";",
                        "  (Space)": " "
                    }
                    actual_separator = sep_map[separator_paste]
                
                with col2:
                    has_header_paste = st.checkbox("First row = headers", value=True, key="header_paste")
                    has_index_paste = st.checkbox("First column = row names", value=False, key="index_paste")
                
                with col3:
                    decimal_paste = st.selectbox("Decimal separator:", [".", ","], key="dec_paste")
                    na_values_paste = st.text_input("Missing value indicators:", value="NA,N/A,NULL,null,#N/A", key="na_paste")
                
                # Preview parsing
                if st.checkbox("🔍 **Preview parsing**", key="preview_parsing"):
                    try:
                        preview_data, detected_sep = _parse_clipboard_data(
                            clipboard_data, actual_separator, decimal_paste, 
                            has_header_paste, has_index_paste, na_values_paste
                        )
                        
                        st.markdown("#### 👀 Data Preview")
                        col_prev1, col_prev2 = st.columns(2)
                        
                        with col_prev1:
                            tab_char = '\t'
                            sep_display = 'Tab' if detected_sep == tab_char else detected_sep
                            st.info(f"**Detected separator**: `{detected_sep}` ({sep_display})")
                            st.info(f"**Shape**: {preview_data.shape[0]} rows × {preview_data.shape[1]} columns")
                        
                        with col_prev2:
                            numeric_cols = preview_data.select_dtypes(include=[np.number]).columns
                            st.info(f"**Numeric variables**: {len(numeric_cols)}")
                            st.info(f"**Missing values**: {preview_data.isnull().sum().sum()}")
                        
                        # Show preview table
                        st.dataframe(preview_data.head(10), use_container_width=True)
                        
                        # Data quality indicators
                        if preview_data.shape[1] > 50:
                            st.success("🧬 **Spectral/High-dimensional data detected** - Perfect for PCA analysis!")
                        elif preview_data.shape[1] > 10:
                            st.success("📊 **Multi-variable dataset** - Great for chemometric analysis!")
                        else:
                            st.info("📋 **Standard dataset** - Ready for analysis!")
                        
                    except Exception as e:
                        st.error(f"❌ **Parse error**: {str(e)}")
                        st.info("💡 Try adjusting the separator or format settings above")
                
                # Load button
                if st.button("🚀 **Load Data from Clipboard**", type="primary", key="load_clipboard"):
                    try:
                        data, detected_sep = _parse_clipboard_data(
                            clipboard_data, actual_separator, decimal_paste, 
                            has_header_paste, has_index_paste, na_values_paste
                        )
                        
                        # Store data
                        dataset_name = f"Clipboard_Data_{pd.Timestamp.now().strftime('%H%M%S')}"
                        st.session_state.current_data = data
                        st.session_state.current_dataset = dataset_name
                        
                        # Save original to transformation history
                        _save_original_to_history(data, dataset_name)
                        
                        # Success message with details
                        st.success(f"🎉 **Data loaded successfully!**")
                        
                        col_success1, col_success2, col_success3, col_success4 = st.columns(4)
                        with col_success1:
                            st.metric("Samples", data.shape[0])
                        with col_success2:
                            st.metric("Variables", data.shape[1])
                        with col_success3:
                            tab_char = '\t'
                            sep_display = 'Tab' if detected_sep == tab_char else detected_sep
                            st.metric("Separator Used", sep_display)
                        with col_success4:
                            numeric_cols = data.select_dtypes(include=[np.number]).columns
                            st.metric("Numeric Vars", len(numeric_cols))
                        
                        # Data preview
                        st.markdown("#### 📊 **Loaded Dataset Preview**")
                        st.dataframe(data.head(10), use_container_width=True)
                        
                        # Auto-suggestions based on data
                        st.markdown("#### 🎯 **Suggested Next Steps**")
                        
                        if data.shape[1] > 20:
                            st.info("🧬 **High-dimensional data** → Try **PCA Analysis** for dimensionality reduction")
                        
                        if data.shape[0] > 100:
                            st.info("📈 **Large dataset** → Consider **Data Transformations** before analysis")
                        
                        if data.isnull().sum().sum() > 0:
                            st.warning(f"⚠️ **{data.isnull().sum().sum()} missing values** detected → Use **Transformations → Handle Missing Data**")
                        
                    except Exception as e:
                        st.error(f"❌ **Loading failed**: {str(e)}")
                        
                        # Troubleshooting suggestions
                        st.markdown("#### 🔧 **Troubleshooting**")
                        st.info("Try these solutions:")
                        st.info("• Change the **separator** setting")
                        st.info("• Check **decimal separator** (. vs ,)")
                        st.info("• Verify **header/index** settings")
                        st.info("• Remove special characters from data")
                        st.info("• Copy smaller data chunks")
            
            else:
                # When no data is pasted, show example
                st.info("👆 **Paste your data above to get started**")
                
                with st.expander("📝 **Example Data Formats**"):
                    st.markdown("**Example 1: Excel/Spreadsheet Data**")
                    st.code("""Sample_ID	Moisture	Protein	Fat	Ash
Sample1	12.5	18.2	3.4	1.2
Sample2	11.8	19.1	3.8	1.1
Sample3	13.2	17.9	3.2	1.3""")
                    
                    st.markdown("**Example 2: Spectral Data (NIR/IR)**")
                    st.code("""Sample_ID	1000nm	1100nm	1200nm	1300nm
Wheat1	0.534	0.623	0.445	0.567
Wheat2	0.612	0.534	0.523	0.589
Wheat3	0.445	0.678	0.599	0.523""")
                    
                    st.markdown("**Example 3: Chemical Analysis Data**")
                    st.code("""Compound,Concentration,pH,Temperature,Yield
Aspirin,95.2,6.8,25.5,87.3
Ibuprofen,98.1,7.2,26.1,91.5
Paracetamol,96.8,6.9,25.8,89.2""")
                
                # Quick start tips
                st.markdown("#### 🚀 **Quick Start Tips**")
                st.info("✅ **Best practices for Copy/Paste:**")
                st.info("• Include column headers for automatic variable naming")
                st.info("• Use consistent missing value indicators (NA, N/A, etc.)")
                st.info("• Check decimal separators match your data (, vs .)")
                st.info("• For large datasets, start with a small sample to test settings")
        
        elif load_method == "Sample Data":
            st.markdown("### Sample Datasets")
            
            sample_datasets = {
                "NIR Spectra (100×1557)": "nir_spectra",
                "Wine Classification (178×13)": "wine_data", 
                "Chemical Mixture (20×5)": "mixture_design",
                "PCA Example (50×10)": "pca_example",
                "Calibration Data (30×8)": "calibration_data"
            }
            
            selected_sample = st.selectbox("Choose sample dataset:", list(sample_datasets.keys()))
            
            if st.button("Load Sample Dataset"):
                # Generate sample data
                if "NIR" in selected_sample:
                    data = pd.DataFrame(np.random.randn(100, 1557), 
                                    columns=[f"Wave_{i}" for i in range(1557)])
                elif "Wine" in selected_sample:
                    data = pd.DataFrame(np.random.randn(178, 13),
                                      columns=[f"Feature_{i}" for i in range(13)])
                    data['Class'] = np.random.randint(1, 4, 178)
                else:
                    rows, cols = map(int, selected_sample.split('(')[1].split(')')[0].split('×'))
                    data = pd.DataFrame(np.random.randn(rows, cols),
                                      columns=[f"Var_{i}" for i in range(cols)])
                
                st.session_state.current_data = data
                st.session_state.current_dataset = selected_sample

                _save_original_to_history(data, selected_sample)
    
                st.success(f"Sample data loaded: {data.shape[0]} rows × {data.shape[1]} columns")
                st.dataframe(data.head(), use_container_width=True)

    # ===== EXPORT DATA TAB =====
    with tab2:
        st.markdown("## Export Data")
        st.markdown("*Equivalent to DH_export_* R scripts*")
        
        if 'current_data' not in st.session_state:
            st.warning("No data loaded. Please load data first.")
        else:
            data = st.session_state.current_data
            
            export_format = st.selectbox(
                "Choose export format:",
                ["CSV", "Tab-delimited TXT", "Excel (XLSX)", "JSON", "DAT (Spectral Data)", "ASC (ASCII Data)", "SAM (NIR Export)"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                include_index = st.checkbox("Include row names/index", value=True)
                include_header = st.checkbox("Include column headers", value=True)
            
            with col2:
                transpose_data = st.checkbox("Transpose data before export", value=False)
                
                if transpose_data:
                    st.info("Data will be transposed: rows↔columns")
            
            # Generate download based on format
            export_data = data.copy()
            if transpose_data:
                export_data = data.T
            
            if export_format == "CSV":
                csv_data = export_data.to_csv(index=include_index, header=include_header)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    f"{st.session_state.current_dataset}.csv",
                    "text/csv"
                )
            elif export_format == "Excel (XLSX)":
                if EXCEL_AVAILABLE:
                    # Create Excel file in memory
                    from io import BytesIO
                    excel_buffer = BytesIO()
                    
                    # Write to Excel with proper formatting
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        # Write main data
                        export_data.to_excel(writer, sheet_name='Data', index=include_index, header=include_header)
                        
                        # Add metadata sheet
                        metadata = pd.DataFrame({
                            'Property': ['Dataset Name', 'Rows', 'Columns', 'Export Date', 'Source'],
                            'Value': [
                                st.session_state.current_dataset,
                                export_data.shape[0],
                                export_data.shape[1],
                                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'CAT Python Data Handling'
                            ]
                        })
                        metadata.to_excel(writer, sheet_name='Metadata', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        "Download Excel (XLSX)",
                        excel_buffer.getvalue(),
                        f"{st.session_state.current_dataset}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("Excel export requires openpyxl package")
                    # Fallback to CSV
                    csv_data = export_data.to_csv(index=include_index, header=include_header)
                    st.download_button(
                        "Download CSV (Alternative)",
                        csv_data,
                        f"{st.session_state.current_dataset}.csv",
                        "text/csv"
                    )
    
    # ===== WORKSPACE TAB - UPDATED =====
    with tab3:
        st.markdown("## Workspace Management")
        st.markdown("*Equivalent to DH_workspace_management.r*")
        
        # Show current active dataset
        if 'current_data' in st.session_state:
            data = st.session_state.current_data
            
            st.markdown("### 📊 Current Active Dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Dataset", st.session_state.current_dataset)
            with col2:
                st.metric("Samples", data.shape[0])
            with col3:
                st.metric("Variables", data.shape[1])
            with col4:
                memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Size", f"{memory_mb:.1f} MB")
            
            # Show if dataset is transformed
            current_name = st.session_state.current_dataset
            if '.' in current_name and not current_name.endswith('_ORIGINAL'):
                # This is a transformed dataset
                transform_type = current_name.split('.')[-1]
                st.info(f"🔬 **Active dataset is transformed**: {transform_type}")
                
                # Find original dataset
                base_name = current_name.split('.')[0]
                original_key = f"{base_name}_ORIGINAL"
                
                if 'transformation_history' in st.session_state and original_key in st.session_state.transformation_history:
                    original_data = st.session_state.transformation_history[original_key]['data']
                    st.info(f"📋 **Original size**: {original_data.shape[0]} × {original_data.shape[1]} → **Current size**: {data.shape[0]} × {data.shape[1]}")
            else:
                st.success("📋 **Dataset**: Original (no transformations applied)")
            
            # Preview
            st.markdown("### Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
        else:
            st.info("Load a dataset to see workspace information")

        # Show Dataset Splits
        if 'split_datasets' in st.session_state and st.session_state.split_datasets:
            st.markdown("---")
            st.markdown("### 📦 Saved Dataset Splits")
            st.info(f"You have {len(st.session_state.split_datasets)} saved dataset splits")
            
            # Group by parent dataset
            splits_by_parent = {}
            for name, info in st.session_state.split_datasets.items():
                parent = info.get('parent', 'Unknown')
                if parent not in splits_by_parent:
                    splits_by_parent[parent] = []
                splits_by_parent[parent].append((name, info))
            
            # Show by group
            for parent, splits in splits_by_parent.items():
                st.markdown(f"#### 📁 From: {parent}")
                
                for name, info in splits:
                    with st.expander(f"**{name}** ({info['n_samples']} samples)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Dataset Info:**")
                            st.write(f"• Type: {info['type']}")
                            st.write(f"• Samples: {info['n_samples']}")
                            st.write(f"• Variables: {info['data'].shape[1]}")
                            st.write(f"• Created: {info['creation_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # Additional info if available
                            if 'selection_method' in info:
                                st.write(f"• Selection: {info['selection_method']}")
                            if 'pc_axes' in info:
                                st.write(f"• PC Axes: {info['pc_axes']}")
                        
                        with col2:
                            st.markdown("**Actions:**")
                            
                            # Load button
                            if st.button(f"📂 Load Dataset", key=f"load_{name}"):
                                st.session_state.current_data = info['data']
                                st.session_state.current_dataset = name
                                st.success(f"✅ Loaded: {name}")
                                st.rerun()
                            
                            # Preview button
                            if st.button(f"👁️ Preview Data", key=f"preview_{name}"):
                                st.dataframe(info['data'].head(5), use_container_width=True)
                            
                            # Delete button
                            if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                                del st.session_state.split_datasets[name]
                                st.success(f"Deleted: {name}")
                                st.rerun()

                            # Download XLSX button
                            if st.button(f"📥 Download XLSX", key=f"download_{name}"):
                                try:
                                    import io
                                    xlsx_buffer = io.BytesIO()
                                    with pd.ExcelWriter(xlsx_buffer, engine='openpyxl') as writer:
                                        info['data'].to_excel(writer, sheet_name='Data', index=True)

                                    xlsx_buffer.seek(0)
                                    st.download_button(
                                        label="💾 Save File",
                                        data=xlsx_buffer.getvalue(),
                                        file_name=f"{name}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f"dl_{name}"
                                    )
                                except Exception as e:
                                    st.error(f"Download failed: {str(e)}")
            
            # Clear all splits button
            st.markdown("---")
            if st.button("🗑️ Clear All Splits", key="clear_all_splits"):
                if st.session_state.get('confirm_clear_splits', False):
                    st.session_state.split_datasets = {}
                    st.session_state.confirm_clear_splits = False
                    st.success("All splits cleared!")
                    st.rerun()
                else:
                    st.session_state.confirm_clear_splits = True
                    st.warning("⚠️ Click again to confirm deletion of all splits")
        
        # Show Transformation History - ENHANCED
        if 'transformation_history' in st.session_state and st.session_state.transformation_history:
            st.markdown("---")
            st.markdown("### 🔬 Transformation History")
            
            # Count different types
            originals = [name for name, info in st.session_state.transformation_history.items() if info.get('transform_type') == 'original']
            transformations = [name for name, info in st.session_state.transformation_history.items() if info.get('transform_type') != 'original']
            
            st.info(f"📊 **Memory**: {len(originals)} original datasets • {len(transformations)} transformations")
            
            # Group transformations by original dataset
            transforms_by_origin = {}
            for name, info in st.session_state.transformation_history.items():
                if info.get('transform_type') == 'original':
                    origin_key = name
                else:
                    origin_key = info.get('original_dataset', 'Unknown')
                
                if origin_key not in transforms_by_origin:
                    transforms_by_origin[origin_key] = {'original': None, 'transformations': []}
                
                if info.get('transform_type') == 'original':
                    transforms_by_origin[origin_key]['original'] = (name, info)
                else:
                    transforms_by_origin[origin_key]['transformations'].append((name, info))
            
            # Display each original dataset and its transformations
            for origin_key, group in transforms_by_origin.items():
                
                # Show original dataset
                if group['original']:
                    orig_name, orig_info = group['original']
                    st.markdown(f"#### 📋 Original: {orig_name.replace('_ORIGINAL', '')}")
                    
                    with st.expander(f"**Original Data** ({orig_info['data'].shape[0]} × {orig_info['data'].shape[1]})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Dataset Info:**")
                            st.write(f"• Status: {orig_info['transform']}")
                            st.write(f"• Shape: {orig_info['data'].shape[0]} × {orig_info['data'].shape[1]}")
                            st.write(f"• Loaded: {orig_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write(f"• Memory: {orig_info['data'].memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                        
                        with col2:
                            st.markdown("**Actions:**")
                            
                            # Load original
                            if st.button(f"📂 Load Original", key=f"load_original_{orig_name}"):
                                st.session_state.current_data = orig_info['data']
                                st.session_state.current_dataset = orig_name.replace('_ORIGINAL', '')
                                st.success(f"✅ Loaded original dataset")
                                st.rerun()
                            
                            # Preview original
                            if st.button(f"👁️ Preview Original", key=f"preview_original_{orig_name}"):
                                st.dataframe(orig_info['data'].head(5), use_container_width=True)
                
                # Show transformations
                if group['transformations']:
                    st.markdown(f"**🔄 Transformations ({len(group['transformations'])}):**")
                    
                    for name, info in group['transformations']:
                        with st.expander(f"**{name.split('.')[-1]}** ({info['data'].shape[0]} × {info['data'].shape[1]})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Transform Info:**")
                                st.write(f"• Type: {info['transform']}")
                                st.write(f"• Shape: {info['data'].shape[0]} × {info['data'].shape[1]}")
                                st.write(f"• Created: {info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                                
                                if 'params' in info and info['params']:
                                    st.write("**Parameters:**")
                                    for key, val in info['params'].items():
                                        st.write(f"• {key}: {val}")
                            
                            with col2:
                                st.markdown("**Actions:**")
                                
                                # Load transformation
                                if st.button(f"📂 Load Dataset", key=f"load_transform_{name}"):
                                    st.session_state.current_data = info['data']
                                    st.session_state.current_dataset = name
                                    st.success(f"✅ Loaded: {name.split('.')[-1]}")
                                    st.rerun()
                                
                                # Preview transformation
                                if st.button(f"👁️ Preview", key=f"preview_transform_{name}"):
                                    st.dataframe(info['data'].head(5), use_container_width=True)
                                
                                # Delete transformation
                                if st.button(f"🗑️ Delete", key=f"delete_transform_{name}"):
                                    del st.session_state.transformation_history[name]
                                    st.success(f"Deleted: {name}")
                                    st.rerun()

                                # Download XLSX button
                                if st.button(f"📥 Download XLSX", key=f"download_transform_{name}"):
                                    try:
                                        import io
                                        xlsx_buffer = io.BytesIO()
                                        with pd.ExcelWriter(xlsx_buffer, engine='openpyxl') as writer:
                                            info['data'].to_excel(writer, sheet_name='Transformed', index=True)

                                        xlsx_buffer.seek(0)
                                        st.download_button(
                                            label="💾 Save File",
                                            data=xlsx_buffer.getvalue(),
                                            file_name=f"{name}_transformed.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key=f"dl_transform_{name}"
                                        )
                                    except Exception as e:
                                        st.error(f"Download failed: {str(e)}")
                
                st.markdown("---")
            
            # Clear all transformations button
            if st.button("🗑️ Clear All Transformations", key="clear_all_transforms"):
                if st.session_state.get('confirm_clear_transforms', False):
                    st.session_state.transformation_history = {}
                    st.session_state.confirm_clear_transforms = False
                    st.success("All transformations cleared!")
                    st.rerun()
                else:
                    st.session_state.confirm_clear_transforms = True
                    st.warning("⚠️ Click again to confirm deletion of all transformations")
        else:
            st.info("📋 No dataset splits or transformations saved yet.")
            st.info("• Use **PCA Analysis → Scores Plots** to create dataset splits")
            st.info("• Use **Transformations** to create preprocessed datasets")
    
    # ===== DATASET OPERATIONS TAB =====
    with tab4:
        st.markdown("## Dataset Operations")
        st.markdown("*Equivalent to DH_dataset_row.r and DH_dataset_column.r*")
        
        if 'current_data' not in st.session_state:
            st.warning("No data loaded. Please load data first.")
        else:
            data = st.session_state.current_data
            
            operation_type = st.selectbox(
                "Choose operation:",
                ["Row Operations", "Column Operations", "Transpose Data"]
            )
            
            if operation_type == "Transpose Data":
                st.markdown("### Transpose Data")
                st.markdown("*Switch between samples×variables and variables×samples format*")
                
                # Show current format
                st.info(f"**Current format**: {data.shape[0]} rows × {data.shape[1]} columns")
                
                # Preview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Data:**")
                    st.dataframe(data.head(3), use_container_width=True)
                
                with col2:
                    st.markdown("**Transposed Preview:**")
                    st.dataframe(data.T.head(3), use_container_width=True)
                
                if st.button("Transpose Data"):
                    transposed_data = data.T
                    st.session_state.current_data = transposed_data
                    st.session_state.current_dataset = f"{st.session_state.current_dataset}_transposed"
                    st.success(f"Data transposed: {transposed_data.shape[0]} × {transposed_data.shape[1]}")
                    st.rerun()
    
    # ===== RANDOMIZE TAB =====
    with tab5:
        st.markdown("## Randomize")
        st.markdown("*Equivalent to DH_randomize.r*")
        
        if 'current_data' not in st.session_state:
            st.warning("No data loaded. Please load data first.")
        else:
            data = st.session_state.current_data
            
            randomize_type = st.selectbox(
                "Randomization type:",
                ["Shuffle rows", "Shuffle columns", "Random sampling"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                seed = st.number_input("Random seed (for reproducibility):", value=42, min_value=0)
                
            with col2:
                if randomize_type == "Random sampling":
                    sample_size = st.number_input("Sample size:", 
                                                min_value=1, 
                                                max_value=len(data), 
                                                value=min(50, len(data)))
            
            if st.button("Apply Randomization"):
                np.random.seed(seed)
                
                if randomize_type == "Shuffle rows":
                    shuffled_data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
                    st.session_state.current_data = shuffled_data
                    st.success("Rows shuffled successfully")
                    
                elif randomize_type == "Shuffle columns":
                    shuffled_cols = np.random.permutation(data.columns)
                    shuffled_data = data[shuffled_cols]
                    st.session_state.current_data = shuffled_data
                    st.success("Columns shuffled successfully")
                    
                elif randomize_type == "Random sampling":
                    sampled_data = data.sample(n=sample_size, random_state=seed)
                    st.session_state.current_data = sampled_data
                    st.success(f"Random sample of {sample_size} rows created")
                
                st.rerun()

    # ===== METADATA MANAGEMENT TAB =====
    with tab6:
        st.markdown("## Metadata Management")
        st.markdown("*Manage metadata/auxiliary variables for chemometric analysis*")
        
        if 'current_data' not in st.session_state:
            st.warning("No data loaded. Please load data first.")
        else:
            data = st.session_state.current_data
            
            st.markdown("### Current Dataset Analysis")
            
            # Automatic dataset analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Variable Classification")
                
                # Automatically classify variables
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Heuristic to identify spectral data
                # Simplified algorithm: if it's a pure number = spectral, otherwise = metadata
                potential_spectral = []
                potential_metadata = []
                
                for col in numeric_cols:
                    col_str = str(col)
                    
                    # Remove spaces and check if it's ONLY a number
                    clean_col = col_str.strip()
                    
                    try:
                        # Try to convert to float
                        num_val = float(clean_col)
                        
                        # If it's a number and looks like wavelength/wavenumber, it's spectral
                        # Typical ranges: 400-2500 nm (NIR), 4000-400 cm-1 (IR), 200-800 nm (UV-Vis)
                        if (200 <= num_val <= 25000):  # Wide range to cover all spectroscopies
                            potential_spectral.append(col)
                        else:
                            # Number outside spectroscopic range = metadata
                            potential_metadata.append(col)
                            
                    except ValueError:
                        # Not convertible to number = definitely metadata
                        # (e.g.: "Moisture", "Protein", "Sample_ID", "% (w/w) of Barley")
                        potential_metadata.append(col)
                
                # Add categorical to metadata
                potential_metadata.extend(categorical_cols)
                
                st.info(f"**Auto-detected:**")
                st.write(f"- Spectral variables: {len(potential_spectral)}")
                st.write(f"- Metadata variables: {len(potential_metadata)}")
                st.write(f"- Total variables: {len(data.columns)}")
            
            with col2:
                st.markdown("#### Manual Variable Selection")
                
                # Manual selection
                st.markdown("**Mark as Metadata:**")
                metadata_vars = st.multiselect(
                    "Select metadata/auxiliary variables:",
                    data.columns.tolist(),
                    default=potential_metadata,
                    key="metadata_selection"
                )
                
                spectral_vars = [col for col in data.columns if col not in metadata_vars]
                
                st.write(f"Spectral/measurement variables: {len(spectral_vars)}")
                st.write(f"Metadata variables: {len(metadata_vars)}")
            
            # Variable preview
            st.markdown("### Variable Preview")
            
            tab_spectral, tab_metadata = st.tabs(["Spectral Data", "Metadata"])
            
            with tab_spectral:
                if spectral_vars:
                    spectral_data = data[spectral_vars]
                    st.markdown(f"**Spectral/Measurement Variables** ({len(spectral_vars)} variables)")
                    
                    if len(spectral_vars) > 20:
                        st.info("Showing first and last 10 variables (large spectral dataset)")
                        preview_cols = spectral_vars[:10] + spectral_vars[-10:]
                        st.dataframe(data[preview_cols].head(10), use_container_width=True)
                    else:
                        st.dataframe(spectral_data.head(10), use_container_width=True)
                    
                    # Statistics for spectral data
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Min Value", f"{spectral_data.min().min():.4f}")
                    with col_stat2:
                        st.metric("Max Value", f"{spectral_data.max().max():.4f}")
                    with col_stat3:
                        st.metric("Range", f"{spectral_data.max().max() - spectral_data.min().min():.4f}")
                else:
                    st.info("No spectral variables selected")
            
            with tab_metadata:
                if metadata_vars:
                    metadata_data = data[metadata_vars]
                    st.markdown(f"**Metadata Variables** ({len(metadata_vars)} variables)")
                    st.dataframe(metadata_data.head(10), use_container_width=True)
                    
                    # Metadata analysis
                    st.markdown("#### Metadata Analysis")
                    for var in metadata_vars:
                        with st.expander(f"Variable: {var}"):
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                st.write(f"**Type:** {data[var].dtype}")
                                st.write(f"**Unique values:** {data[var].nunique()}")
                                st.write(f"**Missing values:** {data[var].isnull().sum()}")
                            
                            with col_info2:
                                if data[var].dtype in ['object', 'category']:
                                    st.write("**Categories:**")
                                    categories = data[var].value_counts().head(10)
                                    for cat, count in categories.items():
                                        st.write(f"- {cat}: {count}")
                                else:
                                    st.write(f"**Min:** {data[var].min()}")
                                    st.write(f"**Max:** {data[var].max()}")
                                    st.write(f"**Mean:** {data[var].mean():.2f}")
                else:
                    st.info("No metadata variables selected")
            
            # Save classification
            st.markdown("### Save Classification")
            
            if st.button("Save Variable Classification"):
                # Save classification to session state
                if 'data_classification' not in st.session_state:
                    st.session_state.data_classification = {}
                
                st.session_state.data_classification[st.session_state.current_dataset] = {
                    'spectral_variables': spectral_vars,
                    'metadata_variables': metadata_vars,
                    'total_samples': len(data),
                    'classification_date': pd.Timestamp.now()
                }
                
                st.success("Variable classification saved!")
                st.info("This classification will be used in PCA Analysis for automatic variable selection.")
            
            # Export options
            st.markdown("### Export Options")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if st.button("Export Spectral Data Only"):
                    if spectral_vars:
                        spectral_only = data[spectral_vars]
                        csv_spectral = spectral_only.to_csv(index=True)
                        st.download_button(
                            "Download Spectral CSV",
                            csv_spectral,
                            f"{st.session_state.current_dataset}_spectral.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("No spectral variables selected")
            
            with col_exp2:
                if st.button("Export Metadata Only"):
                    if metadata_vars:
                        metadata_only = data[metadata_vars]
                        csv_metadata = metadata_only.to_csv(index=True)
                        st.download_button(
                            "Download Metadata CSV",
                            csv_metadata,
                            f"{st.session_state.current_dataset}_metadata.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("No metadata variables selected")

# Initialize workspace path if not exists
if 'workspace_path' not in st.session_state:
    st.session_state.workspace_path = '/workspace'