# coding: utf-8

import numpy as np
from skimage.metrics import structural_similarity as ssim
import warnings
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CRITERIA_WEIGHTS
from src.utils.logger import log_section, log_success

def compute_mse(X_reconstructed, X_target):
    if X_reconstructed.shape != X_target.shape:
        raise ValueError("Shapes incompatibles")
    mse = np.mean((X_reconstructed - X_target) ** 2)
    return float(mse)

def compute_normalized_mse(X_reconstructed, X_target):
    mse = compute_mse(X_reconstructed, X_target)
    variance = np.var(X_target)
    if variance < 1e-10:
        return mse
    return float(mse / variance)

def compute_snr(X_reconstructed, X_target, epsilon=1e-10):
    signal_power = np.var(X_target)
    noise_power = compute_mse(X_reconstructed, X_target)
    if noise_power < epsilon:
        return 100.0
    if signal_power < epsilon:
        return -100.0
    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)

def compute_ssim(X_reconstructed, X_target, data_range=None):
    if data_range is None:
        data_range = X_target.max() - X_target.min()
    if X_reconstructed.ndim == 2:
        ssim_val = ssim(X_target, X_reconstructed, data_range=data_range)
        return float(ssim_val)
    elif X_reconstructed.ndim == 3:
        ssim_values = []
        for i in range(X_reconstructed.shape[2]):
            s = ssim(X_target[:,:,i], X_reconstructed[:,:,i], data_range=data_range)
            ssim_values.append(s)
        return float(np.mean(ssim_values))
    else:
        raise ValueError("Dimension non supportee")

def compute_center_of_mass(image, threshold=None):
    if threshold is not None:
        image_binary = (image > threshold).astype(float)
    else:
        image_binary = image.copy()
    total_intensity = np.sum(image_binary)
    if total_intensity < 1e-10:
        return tuple(np.array(image.shape) / 2)
    if image.ndim == 2:
        y_coords, x_coords = np.indices(image.shape)
        cx = np.sum(x_coords * image_binary) / total_intensity
        cy = np.sum(y_coords * image_binary) / total_intensity
        return (cx, cy)
    elif image.ndim == 3:
        z_coords, y_coords, x_coords = np.indices(image.shape)
        cx = np.sum(x_coords * image_binary) / total_intensity
        cy = np.sum(y_coords * image_binary) / total_intensity
        cz = np.sum(z_coords * image_binary) / total_intensity
        return (cx, cy, cz)
    else:
        raise ValueError("Dimension non supportee")

def compute_localization_error(X_reconstructed, X_target, threshold=0.1):
    t_target = X_target.max() * threshold
    t_recon = X_reconstructed.max() * threshold
    c_target = compute_center_of_mass(X_target, threshold=t_target)
    c_recon = compute_center_of_mass(X_reconstructed, threshold=t_recon)
    distance = np.linalg.norm(np.array(c_recon) - np.array(c_target))
    return float(distance)

def compute_composite_score(X_reconstructed, X_target, weights=None):
    if weights is None:
        weights = CRITERIA_WEIGHTS
    mse = compute_mse(X_reconstructed, X_target)
    snr = compute_snr(X_reconstructed, X_target)
    ssim_val = compute_ssim(X_reconstructed, X_target)
    loc_error = compute_localization_error(X_reconstructed, X_target)
    mse_norm = mse / (np.var(X_target) + 1e-10)
    snr_norm = np.clip(snr / 30.0, 0, 1)
    max_distance = np.linalg.norm(X_target.shape)
    loc_norm = loc_error / max_distance
    composite = (
        weights['mse'] * mse_norm +
        weights['snr'] * (1 - snr_norm) +
        weights['ssim'] * (1 - ssim_val) +
        weights['localization'] * loc_norm
    )
    return {
        'composite_score': float(composite),
        'mse': float(mse),
        'snr': float(snr),
        'ssim': float(ssim_val),
        'localization': float(loc_error)
    }

def evaluate_reconstruction(X_reconstructed, X_target, verbose=True):
    results = {
        'mse': compute_mse(X_reconstructed, X_target),
        'nmse': compute_normalized_mse(X_reconstructed, X_target),
        'snr': compute_snr(X_reconstructed, X_target),
        'ssim': compute_ssim(X_reconstructed, X_target),
        'localization_error': compute_localization_error(X_reconstructed, X_target)
    }
    comp = compute_composite_score(X_reconstructed, X_target)
    results['composite_score'] = comp['composite_score']
    if verbose:
        print("="*70)
        print("EVALUATION DE LA RECONSTRUCTION")
        print("="*70)
        print("MSE                  : {:.6f}".format(results['mse']))
        print("NMSE                 : {:.6f}".format(results['nmse']))
        print("SNR                  : {:.2f} dB".format(results['snr']))
        print("SSIM                 : {:.4f}".format(results['ssim']))
        print("Erreur localisation  : {:.2f} pixels".format(results['localization_error']))
        print("Score composite      : {:.4f}".format(results['composite_score']))
        print("="*70)
    return results

if __name__ == "__main__":
    log_section("TESTS DES CRITERES")
    
    print("\nTest 1 : Reconstruction parfaite")
    X_target = np.random.rand(100, 100)
    X_perfect = X_target.copy()
    results = evaluate_reconstruction(X_perfect, X_target, verbose=True)
    assert results['mse'] < 1e-10
    assert results['ssim'] > 0.99
    log_success("Test 1 PASSED")
    
    print("\nTest 2 : Bruit leger")
    X_noisy = X_target + 0.01 * np.random.randn(100, 100)
    results = evaluate_reconstruction(X_noisy, X_target, verbose=True)
    assert results['snr'] > 20
    log_success("Test 2 PASSED")
    
    print("\nTest 3 : Tumeur decalee")
    X_target_tumor = np.zeros((100, 100))
    X_target_tumor[40:60, 40:60] = 1.0
    X_shifted = np.zeros((100, 100))
    X_shifted[50:70, 50:70] = 1.0
    results = evaluate_reconstruction(X_shifted, X_target_tumor, verbose=True)
    assert results['localization_error'] > 5
    log_success("Test 3 PASSED")
    
    log_section("TOUS LES TESTS PASSES")
    print("\nFichier criteria.py OK!")
