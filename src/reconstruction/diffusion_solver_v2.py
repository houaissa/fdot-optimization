# coding: utf-8
"""
Solveur diffusion FDOT - VERSION CORRIGEE
Correction majeure : Détecteurs avec fonction de sensibilité (pas source lumineuse)
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import warnings

OPTICAL_PROPERTIES = {
    'excitation': {'mu_a': 0.01, 'mu_s_prime': 1.0},
    'emission': {'mu_a': 0.02, 'mu_s_prime': 0.8}
}

A_ROBIN = 1.0

def compute_diffusion_coefficient(mu_a, mu_s_prime):
    return 1.0 / (3.0 * (mu_a + mu_s_prime))

def create_mouse_domain(grid_size=(32, 32), pixel_size=0.5):
    H, W = grid_size
    physical_size = (H * pixel_size, W * pixel_size)
    cy, cx = H // 2, W // 2
    a = W * 0.40
    b = H * 0.30
    y_coords, x_coords = np.ogrid[:H, :W]
    distances = ((x_coords - cx) / a)**2 + ((y_coords - cy) / b)**2
    domain_mask = distances <= 1.0
    return domain_mask, physical_size

def get_boundary_normal(i, j, domain_mask):
    H, W = domain_mask.shape
    if not domain_mask[i, j]:
        return (0, 0), False
    grad_x = 0
    grad_y = 0
    if j > 0:
        grad_x += (1 if domain_mask[i, j-1] else -1)
    if j < W-1:
        grad_x += (1 if domain_mask[i, j+1] else -1)
    if i > 0:
        grad_y += (1 if domain_mask[i-1, j] else -1)
    if i < H-1:
        grad_y += (1 if domain_mask[i+1, j] else -1)
    norm = np.sqrt(grad_x**2 + grad_y**2)
    if norm > 0:
        return (-grad_x/norm, -grad_y/norm), True
    return (0, 0), False

def solve_diffusion_2D_source(source_position, domain_mask, mu_a, mu_s_prime, pixel_size=0.5, source_width=1.0):
    H, W = domain_mask.shape
    N = H * W
    D = compute_diffusion_coefficient(mu_a, mu_s_prime)
    A_matrix = lil_matrix((N, N))
    b_vector = np.zeros(N)
    
    def idx(i, j):
        return i * W + j
    
    dx = pixel_size
    dy = pixel_size
    alpha_x = D / (dx**2)
    alpha_y = D / (dy**2)
    sx, sy = int(source_position[0]), int(source_position[1])
    sigma_source = source_width / pixel_size
    
    for i in range(H):
        for j in range(W):
            k = idx(i, j)
            if not domain_mask[i, j]:
                A_matrix[k, k] = 1.0
                b_vector[k] = 0.0
                continue
            
            normal, on_boundary = get_boundary_normal(i, j, domain_mask)
            
            if on_boundary:
                nx, ny = normal
                robin_coeff_x = abs(nx) / (2 * A_ROBIN * D / dx)
                robin_coeff_y = abs(ny) / (2 * A_ROBIN * D / dy)
                robin_coeff = robin_coeff_x + robin_coeff_y
                A_matrix[k, k] = (2*alpha_x + 2*alpha_y) + mu_a + robin_coeff
                if j > 0 and domain_mask[i, j-1]:
                    A_matrix[k, idx(i, j-1)] = -alpha_x
                if j < W-1 and domain_mask[i, j+1]:
                    A_matrix[k, idx(i, j+1)] = -alpha_x
                if i > 0 and domain_mask[i-1, j]:
                    A_matrix[k, idx(i-1, j)] = -alpha_y
                if i < H-1 and domain_mask[i+1, j]:
                    A_matrix[k, idx(i+1, j)] = -alpha_y
            else:
                A_matrix[k, k] = (2*alpha_x + 2*alpha_y) + mu_a
                if j > 0 and domain_mask[i, j-1]:
                    A_matrix[k, idx(i, j-1)] = -alpha_x
                if j < W-1 and domain_mask[i, j+1]:
                    A_matrix[k, idx(i, j+1)] = -alpha_x
                if i > 0 and domain_mask[i-1, j]:
                    A_matrix[k, idx(i-1, j)] = -alpha_y
                if i < H-1 and domain_mask[i+1, j]:
                    A_matrix[k, idx(i+1, j)] = -alpha_y
            
            dist2 = (i - sy)**2 + (j - sx)**2
            source_value = np.exp(-dist2 / (2 * sigma_source**2))
            source_value = source_value / (2 * np.pi * sigma_source**2 * pixel_size**2)
            b_vector[k] = source_value
    
    A_matrix = A_matrix.tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Phi_flat = spsolve(A_matrix, b_vector)
    Phi = Phi_flat.reshape((H, W))
    Phi = Phi * domain_mask
    return Phi

def compute_detector_sensitivity(detector_position, domain_mask, pixel_size=0.5, sensitivity_decay=5.0):
    H, W = domain_mask.shape
    dx, dy = detector_position
    Phi_m = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            if domain_mask[i, j]:
                dist = np.sqrt((i - dy)**2 + (j - dx)**2)
                Phi_m[i, j] = np.exp(-dist / sensitivity_decay)
    return Phi_m

def compute_sensitivity_matrix_FDOT(source_positions, detector_positions, domain_mask, pixel_size=0.5, verbose=True):
    H, W = domain_mask.shape
    n_sources = len(source_positions)
    n_detectors = len(detector_positions)
    n_measurements = n_sources * n_detectors
    n_voxels = H * W
    mu_a_x = OPTICAL_PROPERTIES['excitation']['mu_a']
    mu_s_x = OPTICAL_PROPERTIES['excitation']['mu_s_prime']
    A = np.zeros((n_measurements, n_voxels))
    measurement_pairs = []
    
    if verbose:
        print(f"\nCalcul matrice A FDOT (CORRIGE)")
        print(f"  {n_sources} sources × {n_detectors} detecteurs = {n_measurements} mesures")
        print(f"  CORRECTION : Detecteurs = fonction sensibilite")
        print("\nEtape 1/2 : Propagation excitation...")
    
    Phi_excitation_cache = {}
    for s_idx, (sx, sy) in enumerate(source_positions):
        Phi_x = solve_diffusion_2D_source((sx, sy), domain_mask, mu_a_x, mu_s_x, pixel_size, source_width=1.0)
        Phi_excitation_cache[s_idx] = Phi_x
        if verbose and (s_idx + 1) % 5 == 0:
            print(f"  [{s_idx+1}/{n_sources}] sources")
    
    if verbose:
        print(f"✓ {n_sources} propagations\n")
        print("Etape 2/2 : Sensibilite detecteurs...")
    
    Phi_detector_cache = {}
    for d_idx, (dx, dy) in enumerate(detector_positions):
        Phi_m = compute_detector_sensitivity((dx, dy), domain_mask, pixel_size, sensitivity_decay=5.0)
        Phi_detector_cache[d_idx] = Phi_m
        if verbose and (d_idx + 1) % 5 == 0:
            print(f"  [{d_idx+1}/{n_detectors}] detecteurs")
    
    if verbose:
        print(f"✓ {n_detectors} sensibilites\n")
    
    measurement_idx = 0
    for s_idx in range(n_sources):
        for d_idx in range(n_detectors):
            Phi_x = Phi_excitation_cache[s_idx]
            Phi_m = Phi_detector_cache[d_idx]
            A_row = Phi_x * Phi_m
            A[measurement_idx, :] = A_row.flatten()
            measurement_pairs.append((s_idx, d_idx))
            measurement_idx += 1
    
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    row_norms[row_norms < 1e-15] = 1.0
    A = A / row_norms
    
    if verbose:
        print(f"✓ Matrice A : {A.shape}")
        print(f"  Normalisation appliquee\n")
    
    return A, measurement_pairs
