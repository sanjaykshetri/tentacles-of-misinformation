"""
Audio Feature Extraction & Preprocessing Module
===============================================

Extracts specialized audio features for deepfake detection:
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Spectral features (centroid, rolloff, bandwidth)
- Prosodic features (pitch, energy)
- Temporal dynamics (delta features)

Used by Phase 3 Step 2 for audio forensics analysis.
"""

import numpy as np
import librosa
import scipy.signal as signal
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract comprehensive audio features for deepfake detection."""
    
    def __init__(self, sr: int = 16000, n_mfcc: int = 13):
        """
        Initialize feature extractor.
        
        Args:
            sr: Sample rate (default 16kHz - standard for speech)
            n_mfcc: Number of MFCC coefficients
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_all_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract complete feature set from audio file.
        
        Returns:
            Dictionary with 35+ audio features
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
        features = {}
        
        # 1. MFCC Features (13 coefficients + deltas)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = mfcc_mean[i]
            features[f'mfcc_{i}_std'] = mfcc_std[i]
        
        # MFCC deltas (velocity of change)
        mfcc_delta = librosa.feature.delta(mfcc)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta_std'] = np.std(mfcc_delta)
        
        # 2. Spectral Features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spec_centroid)
        features['spectral_centroid_std'] = np.std(spec_centroid)
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
        features['spectral_rolloff_std'] = np.std(spec_rolloff)
        
        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spec_bw)
        features['spectral_bandwidth_std'] = np.std(spec_bw)
        
        # 3. Zero-Crossing Rate (voice activity indicator)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. Chroma Features (pitch content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # 5. Energy Features
        energy = np.abs(librosa.stft(y))
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        features['energy_mean'] = energy_mean
        features['energy_std'] = energy_std
        
        # 6. Temporal Features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        features['rms_energy'] = np.sqrt(np.mean(y**2))
        
        # 7. Prosodic Features (Fundamental Frequency estimation)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 0:
            features['f0_mean'] = np.mean(f0_valid)
            features['f0_std'] = np.std(f0_valid)
            features['f0_min'] = np.min(f0_valid)
            features['f0_max'] = np.max(f0_valid)
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_min'] = 0
            features['f0_max'] = 0
        
        features['voicing_ratio'] = np.mean(voiced_flag)
        
        # 8. Spectral Contrast (captures spectral peaks/valleys)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spec_contrast)
        features['spectral_contrast_std'] = np.std(spec_contrast)
        
        return features
    
    def extract_features_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Extract features from multiple audio files."""
        features_list = []
        for idx, path in enumerate(audio_paths):
            if (idx + 1) % 10 == 0:
                print(f"Processing {idx + 1}/{len(audio_paths)}")
            features = self.extract_all_features(path)
            if features is not None:
                features['audio_path'] = path
                features_list.append(features)
        return features_list


class AudioPreprocessor:
    """Handle audio normalization, augmentation, and quality checks."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def normalize_audio(self, y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target loudness (dB)."""
        current_db = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        return y * gain_linear
    
    def remove_silence(self, y: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Remove silence from beginning/end of audio."""
        S = np.abs(librosa.stft(y))
        db = librosa.power_to_db(S, ref=np.max)
        
        # Find frames above threshold
        above_thresh = np.any(db > threshold_db, axis=0)
        
        if np.any(above_thresh):
            first = np.argmax(above_thresh)
            last = len(above_thresh) - np.argmax(above_thresh[::-1])
            
            # Convert frame indices to samples
            hop_length = 512
            first_sample = first * hop_length
            last_sample = last * hop_length
            
            return y[first_sample:last_sample]
        return y
    
    def time_stretch(self, y: np.ndarray, rate: float = 1.1) -> np.ndarray:
        """Time stretch augmentation (simulate speaking rate variation)."""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(self, y: np.ndarray, steps: int = 2) -> np.ndarray:
        """Pitch shift augmentation (simulate voice variation)."""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)
    
    def add_noise(self, y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add Gaussian noise (robustness augmentation)."""
        noise = np.random.randn(len(y)) * noise_factor
        return y + noise
    
    def check_audio_quality(self, path: str) -> Dict[str, bool]:
        """Check audio file for quality issues."""
        try:
            y, sr = librosa.load(path, sr=self.sr)
        except:
            return {'valid': False, 'reason': 'Cannot load'}
        
        checks = {
            'valid': True,
            'duration_ok': len(y) / sr > 0.5,  # At least 0.5 seconds
            'not_silent': np.max(np.abs(y)) > 0.01,  # Not completely silent
            'sample_rate': sr == self.sr
        }
        
        checks['reason'] = 'OK'
        if not checks['duration_ok']:
            checks['reason'] = 'Too short'
        elif not checks['not_silent']:
            checks['reason'] = 'Silent'
        elif not checks['sample_rate']:
            checks['reason'] = f'Wrong sample rate ({sr})'
        
        return checks
    
    def preprocess_pipeline(self, audio_path: str, normalize: bool = True, 
                           remove_silence: bool = True) -> Tuple[np.ndarray, int]:
        """Complete preprocessing pipeline."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        if remove_silence:
            y = self.remove_silence(y)
        
        if normalize:
            y = self.normalize_audio(y)
        
        return y, sr
