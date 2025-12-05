"""
Preprocessing Theory Module
============================

This module provides tools for generating simulated spectral data with various artifacts
and analyzing the effects of different preprocessing methods.

Classes:
    - SimulatedSpectralDataGenerator: Generates synthetic spectral data with controlled artifacts
    - PreprocessingEffectsAnalyzer: Applies various preprocessing transformations

Functions:
    - get_all_simulated_datasets: Convenience function to generate all scenario datasets
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, Tuple, Literal


class SimulatedSpectralDataGenerator:
    """
    Generates simulated spectral data with various artifacts for educational purposes.

    This class creates synthetic spectroscopic data (e.g., Raman/IR spectra) with controlled
    artifacts including baseline shifts, baseline drift, and global intensity variations.

    Attributes:
        n_samples (int): Number of spectra to generate
        n_variables (int): Number of wavenumber points
        wavenumber_min (float): Minimum wavenumber (cm⁻¹)
        wavenumber_max (float): Maximum wavenumber (cm⁻¹)
        noise_level_db (float): Signal-to-noise ratio in dB (range: 20-80)
        random_state (int): Random seed for reproducibility
    """

    def __init__(
        self,
        n_samples: int = 50,
        n_variables: int = 500,
        wavenumber_min: float = 400.0,
        wavenumber_max: float = 1800.0,
        noise_level_db: float = 50.0,
        random_state: int = 42
    ):
        """
        Initialize the spectral data generator.

        Args:
            n_samples: Number of spectra to generate (default: 50)
            n_variables: Number of wavenumber points (default: 500)
            wavenumber_min: Minimum wavenumber in cm⁻¹ (default: 400)
            wavenumber_max: Maximum wavenumber in cm⁻¹ (default: 1800)
            noise_level_db: Signal-to-noise ratio in dB, range 20-80 (default: 50)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.n_samples = n_samples
        self.n_variables = n_variables
        self.wavenumber_min = wavenumber_min
        self.wavenumber_max = wavenumber_max
        self.noise_level_db = np.clip(noise_level_db, 20.0, 80.0)  # Clamp to valid range
        self.random_state = random_state

        # Set random seed
        np.random.seed(self.random_state)

        # Generate wavenumber axis
        self.wavenumbers = np.linspace(wavenumber_min, wavenumber_max, n_variables)

        # Spectral parameters
        self.peak_center = 600.0  # cm⁻¹
        self.baseline_offset = 0.05

    def _generate_gaussian_peak(self, center: float = 600.0, width: float = 50.0,
                                amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a Gaussian peak at specified center position.

        Args:
            center: Peak center position in cm⁻¹
            width: Peak width (standard deviation)
            amplitude: Peak amplitude

        Returns:
            Array with Gaussian peak profile
        """
        return amplitude * np.exp(-0.5 * ((self.wavenumbers - center) / width) ** 2)

    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add Gaussian noise to signal based on SNR.

        Args:
            signal: Input signal array
            snr_db: Signal-to-noise ratio in decibels

        Returns:
            Signal with added noise
        """
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise

    def generate_clean_spectra(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate clean spectral data without artifacts.

        Returns:
            Tuple containing:
                - DataFrame with spectral data (samples × variables)
                - Array with wavenumber values
        """
        # Reset random state for consistency
        np.random.seed(self.random_state)

        spectra = np.zeros((self.n_samples, self.n_variables))

        for i in range(self.n_samples):
            # Generate base spectrum with Gaussian peak
            spectrum = self._generate_gaussian_peak(
                center=self.peak_center,
                width=50.0,
                amplitude=1.0
            )

            # Add baseline offset
            spectrum += self.baseline_offset

            # Add noise
            spectrum = self._add_noise(spectrum, self.noise_level_db)

            spectra[i, :] = spectrum

        # Create DataFrame with column names as wavenumbers
        column_names = [f"{wn:.2f}" for wn in self.wavenumbers]
        df = pd.DataFrame(spectra, columns=column_names)

        return df, self.wavenumbers

    def generate_baseline_shift_spectra(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate spectra with baseline shift artifacts.

        Each spectrum has a different constant baseline offset in the range [0.1, 0.5].

        Returns:
            Tuple containing:
                - DataFrame with spectral data (samples × variables)
                - Array with wavenumber values
        """
        # Reset random state for consistency
        np.random.seed(self.random_state)

        spectra = np.zeros((self.n_samples, self.n_variables))

        for i in range(self.n_samples):
            # Generate base spectrum
            spectrum = self._generate_gaussian_peak(
                center=self.peak_center,
                width=50.0,
                amplitude=1.0
            )

            # Add variable baseline shift [0.1, 0.5]
            baseline_shift = np.random.uniform(0.1, 0.5)
            spectrum += baseline_shift

            # Add noise
            spectrum = self._add_noise(spectrum, self.noise_level_db)

            spectra[i, :] = spectrum

        # Create DataFrame
        column_names = [f"{wn:.2f}" for wn in self.wavenumbers]
        df = pd.DataFrame(spectra, columns=column_names)

        return df, self.wavenumbers

    def generate_baseline_drift_spectra(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate spectra with baseline drift artifacts.

        Each spectrum has a linear drift with slope in range [0.0001, 0.0005].

        Returns:
            Tuple containing:
                - DataFrame with spectral data (samples × variables)
                - Array with wavenumber values
        """
        # Reset random state for consistency
        np.random.seed(self.random_state)

        spectra = np.zeros((self.n_samples, self.n_variables))

        for i in range(self.n_samples):
            # Generate base spectrum
            spectrum = self._generate_gaussian_peak(
                center=self.peak_center,
                width=50.0,
                amplitude=1.0
            )

            # Add baseline offset
            spectrum += self.baseline_offset

            # Add linear drift
            drift_slope = np.random.uniform(0.0001, 0.0005)
            drift = drift_slope * (self.wavenumbers - self.wavenumber_min)
            spectrum += drift

            # Add noise
            spectrum = self._add_noise(spectrum, self.noise_level_db)

            spectra[i, :] = spectrum

        # Create DataFrame
        column_names = [f"{wn:.2f}" for wn in self.wavenumbers]
        df = pd.DataFrame(spectra, columns=column_names)

        return df, self.wavenumbers

    def generate_global_intensity_spectra(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate spectra with global intensity variations.

        Each spectrum is multiplied by a factor in range [0.5, 2.0].

        Returns:
            Tuple containing:
                - DataFrame with spectral data (samples × variables)
                - Array with wavenumber values
        """
        # Reset random state for consistency
        np.random.seed(self.random_state)

        spectra = np.zeros((self.n_samples, self.n_variables))

        for i in range(self.n_samples):
            # Generate base spectrum
            spectrum = self._generate_gaussian_peak(
                center=self.peak_center,
                width=50.0,
                amplitude=1.0
            )

            # Add baseline offset
            spectrum += self.baseline_offset

            # Add noise
            spectrum = self._add_noise(spectrum, self.noise_level_db)

            # Apply global intensity variation [0.5, 2.0]
            intensity_factor = np.random.uniform(0.5, 2.0)
            spectrum *= intensity_factor

            spectra[i, :] = spectrum

        # Create DataFrame
        column_names = [f"{wn:.2f}" for wn in self.wavenumbers]
        df = pd.DataFrame(spectra, columns=column_names)

        return df, self.wavenumbers

    def generate_combined_effects(self) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """
        Generate spectra with ALL THREE EFFECTS combined in every sample.

        This method creates 30 samples where each sample has all three preprocessing
        challenges applied simultaneously, representing realistic spectroscopic data:

        **Effect 1 - Baseline Shift** (varies per sample):
            - Linear increase from 0.50 (sample 0) to 2.50 (sample 29)
            - Formula: shift = 0.50 + (i / 29) * 2.0
            - Represents constant offset variation between samples

        **Effect 2 - Baseline Drift** (increases along spectrum):
            - Drift rate: 0.0015 per wavenumber point
            - Creates sloped baseline increasing with wavenumber index
            - Represents wavelength-dependent baseline effects

        **Effect 3 - Global Intensity** (varies per sample):
            - Linear increase from 0.8 (sample 0) to 3.3 (sample 29)
            - Formula: intensity_scale = 0.8 + (i / 29) * 2.5
            - Represents multiplicative scaling variation

        **Generation Formula for sample i:**
            For each wavenumber point j:
                shift_i = 0.50 + (i/29) * 2.0
                intensity_i = 0.8 + (i/29) * 2.5
                drift_j = 0.0015 * j
                baseline_j = shift_i + drift_j
                peak_j = intensity_i * gaussian(center=800, width=50, amplitude=2.0)
                spectrum[j] = baseline_j + peak_j + noise(σ=0.02)

        Returns:
            Tuple containing:
                - DataFrame with spectral data (30 samples × variables)
                - Array with wavenumber values
                - Dictionary with metadata including sample baselines and intensities
        """
        # Force n_samples to 30 for this method
        n_samples_total = 30

        # Reset random state for consistency
        np.random.seed(self.random_state)

        spectra = np.zeros((n_samples_total, self.n_variables))

        # Storage for metadata
        sample_baselines = []
        sample_intensities = []

        # ========================================================================
        # GENERATE ALL 30 SAMPLES WITH COMBINED EFFECTS
        # ========================================================================
        for i in range(n_samples_total):
            # Set separate seed for reproducibility
            np.random.seed(self.random_state + i)

            # Effect 1: Baseline Shift (varies per sample)
            # Linear progression from 0.50 to 2.50
            baseline_shift = 0.50 + (i / 29) * 2.0
            sample_baselines.append(baseline_shift)

            # Effect 3: Global Intensity (varies per sample)
            # Linear progression from 0.8 to 3.3
            intensity_scale = 0.8 + (i / 29) * 2.5
            sample_intensities.append(intensity_scale)

            # Effect 2: Baseline Drift (increases along spectrum)
            # drift_rate = 0.0015 per wavenumber point
            drift_component = 0.0015 * np.arange(self.n_variables)

            # Generate base Gaussian peak centered at 800 cm⁻¹
            peak = self._generate_gaussian_peak(
                center=800.0,  # Changed from 600 to 800
                width=50.0,
                amplitude=2.0
            )

            # Apply intensity scaling to peak (Effect 3)
            scaled_peak = intensity_scale * peak

            # Build baseline (Effect 1 + Effect 2)
            baseline = baseline_shift + drift_component

            # Combine baseline and scaled peak
            spectrum = baseline + scaled_peak

            # Add noise (σ=0.005 - reduced for cleaner educational visualization)
            noise = np.random.normal(0, 0.005, self.n_variables)
            spectrum += noise

            spectra[i, :] = spectrum

        # Create DataFrame
        column_names = [f"{wn:.2f}" for wn in self.wavenumbers]
        df = pd.DataFrame(spectra, columns=column_names)

        # Create metadata
        metadata = {
            'sample_baselines': sample_baselines,
            'sample_intensities': sample_intensities,
            'all_combined': True,
            'effect_descriptions': {
                'baseline_shift': f'Varies from {sample_baselines[0]:.2f} to {sample_baselines[-1]:.2f}',
                'baseline_drift': 'Rate: 0.0015 per wavenumber point',
                'global_intensity': f'Varies from {sample_intensities[0]:.1f}x to {sample_intensities[-1]:.1f}x'
            }
        }

        return df, self.wavenumbers, metadata


def generate_categorical_diagnostic_dataset(
    n_samples_per_category: int = 10,
    n_variables: int = 500,
    wavenumber_range: Tuple[float, float] = (400.0, 1800.0),
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate a diagnostic spectral dataset with three distinct sample categories.

    This standalone function creates simulated spectral data designed for teaching
    preprocessing method selection. Each category represents a different analytical
    challenge commonly encountered in spectroscopy.

    **Category 1 - "Peak Height Variation" (first n_samples_per_category samples):**
        - Strong baseline (fixed at 0.3)
        - Peak height varies linearly from 1.0 to 3.5
        - Peak width fixed at σ=50
        - Low noise (σ=0.01)
        - **Challenge:** Requires intensity normalization (SNV, MSC, or autoscaling)
        - **Visual indicator:** Different peak amplitudes, same shape

    **Category 2 - "Peak Shape Distortion" (next n_samples_per_category samples):**
        - Fixed baseline (0.3)
        - Peak height fixed at 2.0
        - Peak width varies linearly from σ=30 to σ=80
        - Low noise (σ=0.01)
        - **Challenge:** Requires derivatives to resolve shape differences
        - **Visual indicator:** Same peak height, different curve widths

    **Category 3 - "Noise & Spike Artifacts" (final n_samples_per_category samples):**
        - Variable baseline shift [0.1, 0.5]
        - Fixed peak (height=2.0, σ=50)
        - High noise (σ=0.05) + random spike artifacts (5-8 spikes per spectrum)
        - **Challenge:** Requires smoothing (Savitzky-Golay) + spike removal
        - **Visual indicator:** Noisy profile with random high-intensity spikes

    Args:
        n_samples_per_category: Number of samples per category (default: 10)
        n_variables: Number of wavenumber points (default: 500)
        wavenumber_range: Tuple of (min, max) wavenumber in cm⁻¹ (default: (400, 1800))
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple containing:
            - DataFrame: Spectral data with shape (3*n_samples_per_category, n_variables)
            - Dict: Category metadata including:
                - 'sample_categories': Dict mapping category names to sample indices
                - 'category_labels': List of category label for each sample
                - 'category_descriptions': Dict with preprocessing recommendations
                - 'wavenumbers': Array of wavenumber values
                - 'wavenumber_min': Minimum wavenumber
                - 'wavenumber_max': Maximum wavenumber

    Example:
        >>> data, metadata = generate_categorical_diagnostic_dataset(n_samples_per_category=10)
        >>> print(data.shape)  # (30, 500)
        >>> print(metadata['sample_categories'])
        # {'Category_1_Peak_Height': [0, 1, ..., 9],
        #  'Category_2_Peak_Shape': [10, 11, ..., 19],
        #  'Category_3_Noise_Spikes': [20, 21, ..., 29]}
    """
    # Unpack wavenumber range
    wavenumber_min, wavenumber_max = wavenumber_range

    # Generate wavenumber axis
    wavenumbers = np.linspace(wavenumber_min, wavenumber_max, n_variables)

    # Peak center position
    peak_center = 600.0  # cm⁻¹

    # Total samples
    n_samples_total = 3 * n_samples_per_category

    # Initialize storage
    spectra = np.zeros((n_samples_total, n_variables))
    category_labels = []

    # Helper function to generate Gaussian peak
    def generate_gaussian_peak(center: float, width: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((wavenumbers - center) / width) ** 2)

    # ========================================================================
    # CATEGORY 1: PEAK HEIGHT VARIATION (samples 0 to n_samples_per_category-1)
    # Challenge: Intensity normalization needed
    # ========================================================================
    peak_heights = np.linspace(1.0, 3.5, n_samples_per_category)

    for i in range(n_samples_per_category):
        # Set separate seed for reproducibility
        np.random.seed(seed + 100 + i)

        # Generate base spectrum with VARYING peak height
        spectrum = generate_gaussian_peak(
            center=peak_center,
            width=50.0,
            amplitude=peak_heights[i]  # VARYING AMPLITUDE
        )

        # Fixed strong baseline
        spectrum += 0.3

        # Low noise
        noise = np.random.normal(0, 0.01, n_variables)
        spectrum += noise

        spectra[i, :] = spectrum
        category_labels.append('Category_1_Peak_Height')

    # ========================================================================
    # CATEGORY 2: PEAK SHAPE DISTORTION (samples n to 2n-1)
    # Challenge: Derivatives or baseline correction needed
    # ========================================================================
    peak_widths = np.linspace(30.0, 80.0, n_samples_per_category)

    for i in range(n_samples_per_category):
        idx = n_samples_per_category + i
        # Set separate seed for reproducibility
        np.random.seed(seed + 200 + i)

        # Generate base spectrum with VARYING peak width
        spectrum = generate_gaussian_peak(
            center=peak_center,
            width=peak_widths[i],  # VARYING WIDTH
            amplitude=2.0  # Fixed amplitude
        )

        # Fixed baseline
        spectrum += 0.3

        # Low noise
        noise = np.random.normal(0, 0.01, n_variables)
        spectrum += noise

        spectra[idx, :] = spectrum
        category_labels.append('Category_2_Peak_Shape')

    # ========================================================================
    # CATEGORY 3: NOISE & SPIKE ARTIFACTS (samples 2n to 3n-1)
    # Challenge: Smoothing and spike removal needed
    # ========================================================================
    for i in range(n_samples_per_category):
        idx = 2 * n_samples_per_category + i
        # Set separate seed for reproducibility
        np.random.seed(seed + 300 + i)

        # Generate base spectrum with fixed parameters
        spectrum = generate_gaussian_peak(
            center=peak_center,
            width=50.0,
            amplitude=2.0
        )

        # Variable baseline shift
        baseline_shift = np.random.uniform(0.1, 0.5)
        spectrum += baseline_shift

        # HIGH NOISE
        high_noise = np.random.normal(0, 0.05, n_variables)
        spectrum += high_noise

        # Add random SPIKE ARTIFACTS (5-8 spikes per spectrum)
        n_spikes = np.random.randint(5, 9)
        spike_positions = np.random.choice(n_variables, size=n_spikes, replace=False)
        spike_intensities = np.random.uniform(0.5, 1.5, size=n_spikes)

        for pos, intensity in zip(spike_positions, spike_intensities):
            spectrum[pos] += intensity

        spectra[idx, :] = spectrum
        category_labels.append('Category_3_Noise_Spikes')

    # Create DataFrame
    column_names = [f"{wn:.2f}" for wn in wavenumbers]
    df = pd.DataFrame(spectra, columns=column_names)

    # Create category metadata
    category_metadata = {
        'sample_categories': {
            'Category_1_Peak_Height': list(range(0, n_samples_per_category)),
            'Category_2_Peak_Shape': list(range(n_samples_per_category, 2 * n_samples_per_category)),
            'Category_3_Noise_Spikes': list(range(2 * n_samples_per_category, 3 * n_samples_per_category))
        },
        'category_labels': category_labels,
        'category_descriptions': {
            'Category_1_Peak_Height': 'Peak height variation [1.0-3.5] - Requires intensity normalization (SNV/MSC/Autoscaling)',
            'Category_2_Peak_Shape': 'Peak width variation [30-80] - Requires derivatives or baseline correction',
            'Category_3_Noise_Spikes': 'High noise + spike artifacts - Requires smoothing (Savitzky-Golay) and spike removal'
        },
        'wavenumbers': wavenumbers,
        'wavenumber_min': wavenumber_min,
        'wavenumber_max': wavenumber_max
    }

    return df, category_metadata


class PreprocessingEffectsAnalyzer:
    """
    Analyzes the effects of various preprocessing methods on spectral data.

    This class provides methods to apply different preprocessing transformations
    commonly used in spectroscopy, including SNV, derivatives, and Savitzky-Golay filtering.

    Attributes:
        data (pd.DataFrame): Input spectral data (samples × variables)
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with spectral data.

        Args:
            data: DataFrame containing spectral data (samples × variables)
        """
        self.data = data.copy()
        self.column_names = data.columns.tolist()

    def snv_transform(self) -> pd.DataFrame:
        """
        Apply Standard Normal Variate (SNV) transformation.

        SNV performs row-wise autoscaling (centering and scaling each spectrum
        to zero mean and unit variance). This corrects for multiplicative and
        additive scatter effects.

        Returns:
            DataFrame with SNV-transformed data
        """
        data_array = self.data.values

        # Row-wise centering and scaling
        row_means = np.mean(data_array, axis=1, keepdims=True)
        row_stds = np.std(data_array, axis=1, keepdims=True, ddof=1)

        # Avoid division by zero
        row_stds[row_stds == 0] = 1.0

        snv_data = (data_array - row_means) / row_stds

        # Preserve DataFrame structure
        return pd.DataFrame(snv_data, columns=self.column_names, index=self.data.index)

    def first_derivative(self) -> pd.DataFrame:
        """
        Apply first derivative using simple finite differences.

        Computes the first derivative using numpy.diff along the spectral axis.
        The resulting data has one fewer variable than the input.

        Returns:
            DataFrame with first derivative data
        """
        data_array = self.data.values

        # Compute first derivative
        deriv = np.diff(data_array, n=1, axis=1)

        # Update column names (one fewer column)
        new_columns = self.column_names[:-1]

        return pd.DataFrame(deriv, columns=new_columns, index=self.data.index)

    def first_derivative_savitzky_golay(self, window: int = 11, polyorder: int = 3) -> pd.DataFrame:
        """
        Apply first derivative using Savitzky-Golay filter.

        The Savitzky-Golay filter fits successive sub-sets of adjacent data points
        with a low-degree polynomial by linear least squares, providing smoothed
        derivative estimates.

        Args:
            window: Length of the filter window (must be odd, default: 11)
            polyorder: Order of polynomial used to fit samples (default: 3)

        Returns:
            DataFrame with Savitzky-Golay first derivative data
        """
        data_array = self.data.values

        # Apply Savitzky-Golay filter for first derivative
        deriv = savgol_filter(data_array, window_length=window, polyorder=polyorder,
                             deriv=1, axis=1, mode='nearest')

        # Preserve DataFrame structure
        return pd.DataFrame(deriv, columns=self.column_names, index=self.data.index)

    def second_derivative(self) -> pd.DataFrame:
        """
        Apply second derivative using simple finite differences.

        Computes the second derivative by applying diff twice.
        The resulting data has two fewer variables than the input.

        Returns:
            DataFrame with second derivative data
        """
        data_array = self.data.values

        # Compute second derivative
        deriv = np.diff(data_array, n=2, axis=1)

        # Update column names (two fewer columns)
        new_columns = self.column_names[:-2]

        return pd.DataFrame(deriv, columns=new_columns, index=self.data.index)

    def second_derivative_savitzky_golay(self, window: int = 15, polyorder: int = 3) -> pd.DataFrame:
        """
        Apply second derivative using Savitzky-Golay filter.

        Args:
            window: Length of the filter window (must be odd, default: 15)
            polyorder: Order of polynomial used to fit samples (default: 3)

        Returns:
            DataFrame with Savitzky-Golay second derivative data
        """
        data_array = self.data.values

        # Apply Savitzky-Golay filter for second derivative
        deriv = savgol_filter(data_array, window_length=window, polyorder=polyorder,
                             deriv=2, axis=1, mode='nearest')

        # Preserve DataFrame structure
        return pd.DataFrame(deriv, columns=self.column_names, index=self.data.index)

    def apply_preprocessing(
        self,
        method: Literal['snv', 'first_derivative', 'first_derivative_sg',
                       'second_derivative', 'second_derivative_sg']
    ) -> Tuple[pd.DataFrame, str]:
        """
        Apply a specified preprocessing method.

        Args:
            method: Preprocessing method to apply. Options:
                - 'snv': Standard Normal Variate
                - 'first_derivative': First derivative (finite differences)
                - 'first_derivative_sg': First derivative (Savitzky-Golay)
                - 'second_derivative': Second derivative (finite differences)
                - 'second_derivative_sg': Second derivative (Savitzky-Golay)

        Returns:
            Tuple containing:
                - DataFrame with processed data
                - String description of the method applied
        """
        method_map = {
            'snv': (
                self.snv_transform,
                "Standard Normal Variate (SNV) - Row autoscaling for scatter correction"
            ),
            'first_derivative': (
                self.first_derivative,
                "First Derivative (Finite Differences) - Baseline correction"
            ),
            'first_derivative_sg': (
                lambda: self.first_derivative_savitzky_golay(window=11, polyorder=3),
                "First Derivative (Savitzky-Golay, window=11, polyorder=3) - Smoothed baseline correction"
            ),
            'second_derivative': (
                self.second_derivative,
                "Second Derivative (Finite Differences) - Enhanced peak resolution"
            ),
            'second_derivative_sg': (
                lambda: self.second_derivative_savitzky_golay(window=15, polyorder=3),
                "Second Derivative (Savitzky-Golay, window=15, polyorder=3) - Smoothed peak resolution"
            ),
        }

        if method not in method_map:
            raise ValueError(
                f"Unknown method '{method}'. Available methods: {list(method_map.keys())}"
            )

        transform_func, description = method_map[method]
        processed_data = transform_func()

        return processed_data, description


def get_all_simulated_datasets(n_samples: int = 50, noise_level_db: float = 50.0) -> Dict[str, Dict]:
    """
    Generate all simulated spectral datasets for preprocessing demonstrations.

    This convenience function creates five different spectral datasets, each demonstrating
    different artifacts commonly encountered in spectroscopy.

    Args:
        n_samples: Number of spectra to generate per dataset (default: 50)
        noise_level_db: Signal-to-noise ratio in dB, range 20-80 (default: 50)

    Returns:
        Dictionary with keys 'clean', 'baseline_shift', 'baseline_drift',
        'global_intensity', and 'combined_effects'. Each entry contains:
            - 'data': DataFrame with spectral data
            - 'wavenumbers': Array of wavenumber values
            - 'wavenumber_min': Minimum wavenumber
            - 'wavenumber_max': Maximum wavenumber
            - 'effect_type': String describing the artifact type
            - 'description': Detailed description of the artifact
    """
    # Initialize generator
    generator = SimulatedSpectralDataGenerator(
        n_samples=n_samples,
        n_variables=500,
        wavenumber_min=400.0,
        wavenumber_max=1800.0,
        noise_level_db=noise_level_db,
        random_state=42
    )

    # Generate all datasets
    clean_data, clean_wn = generator.generate_clean_spectra()
    baseline_shift_data, baseline_shift_wn = generator.generate_baseline_shift_spectra()
    baseline_drift_data, baseline_drift_wn = generator.generate_baseline_drift_spectra()
    global_intensity_data, global_intensity_wn = generator.generate_global_intensity_spectra()
    combined_effects_data, combined_effects_wn, category_metadata = generator.generate_combined_effects()

    # Package results
    datasets = {
        'clean': {
            'data': clean_data,
            'wavenumbers': clean_wn,
            'wavenumber_min': generator.wavenumber_min,
            'wavenumber_max': generator.wavenumber_max,
            'effect_type': 'Clean (Reference)',
            'description': f'Clean spectral data with minimal artifacts. Gaussian peak at 600 cm⁻¹ with baseline offset and noise (SNR={noise_level_db:.0f}dB).'
        },
        'baseline_shift': {
            'data': baseline_shift_data,
            'wavenumbers': baseline_shift_wn,
            'wavenumber_min': generator.wavenumber_min,
            'wavenumber_max': generator.wavenumber_max,
            'effect_type': 'Baseline Shift',
            'description': 'Spectra with constant baseline shifts varying between samples (range: 0.1-0.5). Common in spectroscopy due to sample thickness or scattering variations.'
        },
        'baseline_drift': {
            'data': baseline_drift_data,
            'wavenumbers': baseline_drift_wn,
            'wavenumber_min': generator.wavenumber_min,
            'wavenumber_max': generator.wavenumber_max,
            'effect_type': 'Baseline Drift',
            'description': 'Spectra with linear baseline drift (slope: 0.0001-0.0005). Simulates wavelength-dependent baseline effects common in IR/Raman spectroscopy.'
        },
        'global_intensity': {
            'data': global_intensity_data,
            'wavenumbers': global_intensity_wn,
            'wavenumber_min': generator.wavenumber_min,
            'wavenumber_max': generator.wavenumber_max,
            'effect_type': 'Global Intensity Variation',
            'description': 'Spectra with varying overall intensities (factor: 0.5-2.0). Represents multiplicative effects from concentration or pathlength differences.'
        },
        'combined_effects': {
            'data': combined_effects_data,
            'wavenumbers': combined_effects_wn,
            'wavenumber_min': generator.wavenumber_min,
            'wavenumber_max': generator.wavenumber_max,
            'effect_type': 'Combined Effects (Realistic)',
            'description': 'Realistic spectral data with ALL THREE effects combined in every sample: (1) Baseline Shift - varies from 0.50 to 2.50 across samples, (2) Baseline Drift - increases at rate 0.0015 per wavenumber point, (3) Global Intensity - varies from 0.8x to 3.3x across samples. Peak centered at 800 cm⁻¹ with noise σ=0.02.',
            'combined_metadata': category_metadata
        }
    }

    return datasets


# Example usage
if __name__ == "__main__":
    # Generate all datasets
    datasets = get_all_simulated_datasets(n_samples=50)

    print("Generated Simulated Datasets:")
    print("=" * 60)
    for key, dataset_info in datasets.items():
        print(f"\n{dataset_info['effect_type']}:")
        print(f"  Description: {dataset_info['description']}")
        print(f"  Data shape: {dataset_info['data'].shape}")
        print(f"  Wavenumber range: {dataset_info['wavenumber_min']:.1f} - {dataset_info['wavenumber_max']:.1f} cm⁻¹")

    # Demonstrate preprocessing
    print("\n\nPreprocessing Methods Available:")
    print("=" * 60)

    # Take baseline shift data as example
    example_data = datasets['baseline_shift']['data']
    analyzer = PreprocessingEffectsAnalyzer(example_data)

    methods = ['snv', 'first_derivative', 'first_derivative_sg',
               'second_derivative', 'second_derivative_sg']

    for method in methods:
        processed, description = analyzer.apply_preprocessing(method)
        print(f"\n{method}:")
        print(f"  {description}")
        print(f"  Output shape: {processed.shape}")
