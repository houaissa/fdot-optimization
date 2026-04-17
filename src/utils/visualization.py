"""
Utilitaires pour visualiser les données FDOT

Auteur : Hayat OUAISSA
Date : Mars 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage (pour serveur)
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import FIGURES_DIR

def show_image(image, title="Image", cmap='viridis', save_path=None):
    """Affiche/sauvegarde une image 2D"""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap)
    plt.colorbar(label='Intensité')
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure sauvegardée : {save_path}")
    
    plt.close()

def compare_images(images, titles, cmap='viridis', save_path=None):
    """Compare plusieurs images côte à côte"""
    n_images = len(images)
    
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 4))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        im = axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparaison sauvegardée : {save_path}")
    
    plt.close()

if __name__ == "__main__":
    print("="*70)
    print("TEST DU MODULE VISUALIZATION")
    print("="*70)
    
    # Test avec données synthétiques
    X_test = np.random.rand(100, 100)
    
    print("\nTest : Sauvegarde d'une image de test")
    show_image(X_test, title="Image de test", save_path="results/figures/test_viz.png")
    
    print("✅ Module visualization prêt !")
