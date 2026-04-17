"""
Utilitaires pour charger les données FDOT

Auteur : Hayat OUAISSA
Date : Mars 2026
"""

import numpy as np
from pathlib import Path
import warnings

# Importer la config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_RAW, SOURCE_FILE_PATTERN, TARGET_FILE_NAME

# =============================================================================
# CHARGEMENT DES SOURCES
# =============================================================================

def load_source_image(source_idx, data_dir=None):
    """
    Charge l'image Y pour une source donnée.
    
    Args:
        source_idx (int): Indice de la source (1, 2, 3, ...)
        data_dir (Path): Dossier contenant les données
    
    Returns:
        np.ndarray: Image Y de la source
    """
    if data_dir is None:
        data_dir = DATA_RAW
    
    filename = data_dir / SOURCE_FILE_PATTERN.format(source_idx)
    
    if not filename.exists():
        raise FileNotFoundError(f"Fichier source non trouvé : {filename}")
    
    try:
        Y = np.load(filename)
    except Exception as e:
        raise IOError(f"Erreur chargement {filename}: {e}")
    
    return Y


def load_all_sources(n_sources=None, data_dir=None, verbose=True):
    """
    Charge toutes les images sources.
    
    Args:
        n_sources (int): Nombre de sources à charger
        data_dir (Path): Dossier contenant les données
        verbose (bool): Afficher progression
    
    Returns:
        list: Liste des images Y
    """
    if data_dir is None:
        data_dir = DATA_RAW
    
    all_sources = []
    
    if n_sources is None:
        if verbose:
            print("Détection automatique du nombre de sources...")
        idx = 1
        while True:
            try:
                Y = load_source_image(idx, data_dir)
                all_sources.append(Y)
                idx += 1
            except FileNotFoundError:
                break
        n_sources = len(all_sources)
        if verbose:
            print(f"✅ {n_sources} sources détectées")
    else:
        if verbose:
            print(f"Chargement de {n_sources} sources...")
        
        for idx in range(1, n_sources + 1):
            Y = load_source_image(idx, data_dir)
            all_sources.append(Y)
            
            if verbose and idx % 10 == 0:
                print(f"  [{idx}/{n_sources}]")
        
        if verbose:
            print(f"✅ {n_sources} sources chargées")
    
    return all_sources


def load_target(data_dir=None):
    """
    Charge la vérité terrain X_target.
    
    Args:
        data_dir (Path): Dossier contenant les données
    
    Returns:
        np.ndarray: X_target
    """
    if data_dir is None:
        data_dir = DATA_RAW
    
    filename = data_dir / TARGET_FILE_NAME
    
    if not filename.exists():
        raise FileNotFoundError(f"Fichier target non trouvé : {filename}")
    
    try:
        X_target = np.load(filename)
    except Exception as e:
        raise IOError(f"Erreur chargement {filename}: {e}")
    
    return X_target


def combine_sources(source_indices, all_sources=None, data_dir=None):
    """
    Combine plusieurs sources en sommant leurs images.
    
    Args:
        source_indices (list): Liste des indices de sources
        all_sources (list): Liste préchargée de toutes les sources
        data_dir (Path): Dossier des données
    
    Returns:
        np.ndarray: Image Y combinée
    """
    if len(source_indices) == 0:
        raise ValueError("source_indices ne peut pas être vide")
    
    if all_sources is not None:
        Y_list = [all_sources[i-1] for i in source_indices]
    else:
        Y_list = [load_source_image(i, data_dir) for i in source_indices]
    
    Y_combined = np.sum(Y_list, axis=0)
    
    return Y_combined


def get_data_info(data_dir=None):
    """
    Récupère les informations sur les données disponibles.
    
    Returns:
        dict: Infos sur les données
    """
    if data_dir is None:
        data_dir = DATA_RAW
    
    info = {}
    
    try:
        all_Y = load_all_sources(data_dir=data_dir, verbose=False)
        info['n_sources'] = len(all_Y)
        
        if len(all_Y) > 0:
            info['image_shape'] = all_Y[0].shape
            info['image_dtype'] = all_Y[0].dtype
        else:
            info['image_shape'] = None
            info['image_dtype'] = None
    except:
        info['n_sources'] = 0
        info['image_shape'] = None
        info['image_dtype'] = None
    
    try:
        X_target = load_target(data_dir)
        info['target_exists'] = True
        info['target_shape'] = X_target.shape
    except FileNotFoundError:
        info['target_exists'] = False
        info['target_shape'] = None
    
    return info


if __name__ == "__main__":
    print("="*70)
    print("TEST DU MODULE DATA_LOADER")
    print("="*70)
    
    print("\nTest : Récupération des infos")
    try:
        info = get_data_info()
        print(f"✅ Nombre de sources : {info['n_sources']}")
        print(f"✅ Shape images      : {info['image_shape']}")
        print(f"✅ Target existe     : {info['target_exists']}")
        if info['target_exists']:
            print(f"✅ Shape target      : {info['target_shape']}")
    except Exception as e:
        print(f"⚠️  Pas encore de données : {e}")
    
    print("\n" + "="*70)
    print("Module data_loader OK !")
    print("="*70)
