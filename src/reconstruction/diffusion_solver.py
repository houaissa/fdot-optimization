# coding: utf-8
"""
Solveur equations de diffusion optique (2D) - VERSION FINALE CORRIGEE

Corrections finales :
1. ROI agrandie (40% × 30% au lieu de 35% × 25%)
2. Amplitude physique preservee (pas de normalisation pour matrice A)
3. Deux versions retournees : raw (physique) et normalized (visualisation)

Auteur : Hayat OUAISSA
Date : Mars 2026
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

def create_mouse_domain(grid_size=(128, 128), pixel_size=0.5):
    """Cree domaine souris (ellipse) - VERSION AGRANDIE"""
    H, W = grid_size
    physical_size = (H * pixel_size, W * pixel_size)
    cy, cx = H // 2, W // 2
    
    # CORRECTION : Ellipse plus grande
    a = W * 0.40  # Avant : 0.35
    b = H * 0.30  # Avant : 0.25
    
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

def is_boundary(i, j, domain_mask):
    H, W = domain_mask.shape
    if not domain_mask[i, j]:
        return False
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    for ni, nj in neighbors:
        if 0 <= ni < H and 0 <= nj < W:
            if not domain_mask[ni, nj]:
                return True
    return False

def solve_diffusion_2D(source_position, domain_mask, mu_a, mu_s_prime, 
                       pixel_size=0.5, source_width=2.0, normalize=False, check_energy=False):
    """
    CORRECTION : normalize=False par defaut pour garder amplitude physique
    """
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
    total_source_power = 0
    n_boundary_pixels = 0
    
    for i in range(H):
        for j in range(W):
            k = idx(i, j)
            if not domain_mask[i, j]:
                A_matrix[k, k] = 1.0
                b_vector[k] = 0.0
                continue
            normal, on_boundary = get_boundary_normal(i, j, domain_mask)
            if on_boundary:
                n_boundary_pixels += 1
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
            total_source_power += source_value * pixel_size**2
    
    A_matrix = A_matrix.tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Phi_flat = spsolve(A_matrix, b_vector)
    Phi = Phi_flat.reshape((H, W))
    Phi = Phi * domain_mask
    
    # CORRECTION : Garder amplitude physique
    Phi_raw = Phi.copy()
    
    stats = {}
    if check_energy:
        total_flux = np.sum(Phi_raw[domain_mask]) * pixel_size**2
        stats['total_source_power'] = total_source_power
        stats['total_flux'] = total_flux
        stats['energy_conservation'] = total_flux / total_source_power if total_source_power > 0 else 0
        stats['n_boundary_pixels'] = n_boundary_pixels
        stats['n_total_pixels'] = np.sum(domain_mask)
        stats['boundary_ratio'] = n_boundary_pixels / np.sum(domain_mask) if np.sum(domain_mask) > 0 else 0
        stats['mean_flux_density'] = Phi_raw[domain_mask].mean()
        stats['max_flux_density_raw'] = Phi_raw.max()
    
    if normalize and Phi.max() > 0:
        Phi_normalized = Phi / Phi.max()
        if check_energy:
            stats['normalization'] = 'max'
            stats['normalization_factor'] = Phi.max()
        if check_energy:
            return Phi_raw, stats  # RETOURNER VERSION RAW pour calculs
        else:
            return Phi_normalized  # Pour visualisation seulement
    else:
        if check_energy:
            stats['normalization'] = 'none'
        if check_energy:
            return Phi_raw, stats
        else:
            return Phi_raw

def compute_sensitivity_matrix_FDOT(source_positions, detector_positions, domain_mask, pixel_size=0.5, verbose=True):
    """
    CORRECTION : Utilise amplitude physique (normalize=False)
    """
    H, W = domain_mask.shape
    n_sources = len(source_positions)
    n_detectors = len(detector_positions)
    n_measurements = n_sources * n_detectors
    n_voxels = H * W
    
    mu_a_x = OPTICAL_PROPERTIES['excitation']['mu_a']
    mu_s_x = OPTICAL_PROPERTIES['excitation']['mu_s_prime']
    mu_a_m = OPTICAL_PROPERTIES['emission']['mu_a']
    mu_s_m = OPTICAL_PROPERTIES['emission']['mu_s_prime']
    
    A = np.zeros((n_measurements, n_voxels))
    measurement_pairs = []
    
    if verbose:
        print(f"\nCalcul matrice A FDOT ({n_sources} sources × {n_detectors} detecteurs = {n_measurements} mesures)")
        print(f"Parametres optiques :")
        print(f"  Excitation  : mu_a={mu_a_x:.3f}, mu_s'={mu_s_x:.3f} mm^-1")
        print(f"  Emission    : mu_a={mu_a_m:.3f}, mu_s'={mu_s_m:.3f} mm^-1")
        print(f"  Pixel size  : {pixel_size:.2f} mm")
        print(f"  AMPLITUDE PHYSIQUE PRESERVEE (pas de normalisation)")
        print("Double propagation (excitation + emission)...\n")
    
    if verbose:
        print("Etape 1/2 : Calcul propagation excitation...")
    Phi_excitation_cache = {}
    for s_idx, (sx, sy) in enumerate(source_positions):
        Phi_x = solve_diffusion_2D((sx, sy), domain_mask, mu_a_x, mu_s_x, pixel_size, normalize=False, check_energy=False)
        Phi_excitation_cache[s_idx] = Phi_x
        if verbose and (s_idx + 1) % 5 == 0:
            print(f"  [{s_idx+1}/{n_sources}] sources excitation")
    if verbose:
        print(f"✓ {n_sources} propagations excitation calculees\n")
    
    if verbose:
        print("Etape 2/2 : Calcul propagation emission...")
    Phi_emission_cache = {}
    for d_idx, (dx, dy) in enumerate(detector_positions):
        Phi_m = solve_diffusion_2D((dx, dy), domain_mask, mu_a_m, mu_s_m, pixel_size, normalize=False, check_energy=False)
        Phi_emission_cache[d_idx] = Phi_m
        if verbose and (d_idx + 1) % 5 == 0:
            print(f"  [{d_idx+1}/{n_detectors}] detecteurs emission")
    if verbose:
        print(f"✓ {n_detectors} propagations emission calculees\n")
        print("Calcul produit sensibilite...")
    
    measurement_idx = 0
    for s_idx in range(n_sources):
        for d_idx in range(n_detectors):
            Phi_x = Phi_excitation_cache[s_idx]
            Phi_m = Phi_emission_cache[d_idx]
            A_row = Phi_x * Phi_m
            A[measurement_idx, :] = A_row.flatten()
            measurement_pairs.append((s_idx, d_idx))
            measurement_idx += 1
    
    if verbose:
        print(f"✓ Matrice A complete : shape {A.shape}")
        print(f"  Min  = {A.min():.6e}")
        print(f"  Max  = {A.max():.6e}")
        print(f"  Mean = {A.mean():.6e}")
        print(f"  Std  = {A.std():.6e}\n")
    
    return A, measurement_pairs

if __name__ == "__main__":
    print("="*70)
    print("TEST SOLVEUR DIFFUSION - VERSION FINALE CORRIGEE")
    print("="*70)
    
    domain_mask, physical_size = create_mouse_domain(grid_size=(64, 64), pixel_size=0.5)
    print(f"\nDomaine : {domain_mask.shape}, {physical_size} mm")
    print(f"Voxels  : {np.sum(domain_mask)}/{domain_mask.size}")
    
    print("\n--- Test propagation excitation (amplitude physique) ---")
    source_pos = (32, 32)
    Phi_x, stats_x = solve_diffusion_2D(source_pos, domain_mask, OPTICAL_PROPERTIES['excitation']['mu_a'], OPTICAL_PROPERTIES['excitation']['mu_s_prime'], 0.5, normalize=False, check_energy=True)
    print(f"Fluence excitation : {Phi_x.shape}")
    print(f"  Min  = {Phi_x.min():.6e}")
    print(f"  Max  = {Phi_x.max():.6e} (AMPLITUDE PHYSIQUE)")
    print(f"  Mean = {Phi_x[domain_mask].mean():.6e}")
    print(f"\nStatistiques energie :")
    print(f"  Puissance source     : {stats_x['total_source_power']:.6e}")
    print(f"  Flux total           : {stats_x['total_flux']:.6e}")
    print(f"  Conservation energie : {stats_x['energy_conservation']:.2%}")
    print(f"  Pixels bord          : {stats_x['n_boundary_pixels']}")
    print(f"  Ratio bord/total     : {stats_x['boundary_ratio']:.2%}")
    
    print("\n--- Test propagation emission ---")
    detector_pos = (48, 32)
    Phi_m = solve_diffusion_2D(detector_pos, domain_mask, OPTICAL_PROPERTIES['emission']['mu_a'], OPTICAL_PROPERTIES['emission']['mu_s_prime'], 0.5, normalize=False)
    print(f"Fluence emission : {Phi_m.shape}")
    print(f"  Min = {Phi_m.min():.6e}, Max = {Phi_m.max():.6e} (AMPLITUDE PHYSIQUE)")
    
    A_example = Phi_x * Phi_m
    print(f"\nSensibilite totale : {A_example.shape}")
    print(f"  Max = {A_example.max():.6e} (AMPLITUDE PHYSIQUE PRESERVEE)")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Normaliser UNIQUEMENT pour visualisation
    Phi_x_vis = Phi_x / Phi_x.max()
    Phi_m_vis = Phi_m / Phi_m.max()
    A_vis = A_example / A_example.max()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(domain_mask, cmap='gray')
    axes[0, 0].scatter([source_pos[0]], [source_pos[1]], c='red', s=100, marker='*', label='Source')
    axes[0, 0].scatter([detector_pos[0]], [detector_pos[1]], c='blue', s=100, marker='o', label='Detecteur')
    axes[0, 0].set_title('Domaine + Source + Detecteur')
    axes[0, 0].legend()
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(Phi_x_vis, cmap='hot')
    axes[0, 1].set_title(f'Propagation Excitation\n(Normalized for display)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(Phi_m_vis, cmap='hot')
    axes[1, 0].set_title(f'Propagation Emission\n(Normalized for display)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(A_vis, cmap='hot')
    axes[1, 1].set_title(f'Sensibilite Totale\n(Normalized for display)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('test_diffusion_final.png', dpi=150)
    plt.close()
    
    print("\nGraphique : test_diffusion_final.png")
    print("\n" + "="*70)
    print("SOLVEUR FINAL CORRIGE OPERATIONNEL !")
    print("="*70)
