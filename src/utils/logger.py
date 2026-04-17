"""
Utilitaires pour logger proprement

Auteur : Hayat OUAISSA
Date : Mars 2026
"""

from datetime import datetime

def log_info(message):
    """Affiche un message d'information"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ℹ️  {message}")

def log_success(message):
    """Affiche un message de succès"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ✅ {message}")

def log_warning(message):
    """Affiche un avertissement"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ⚠️  {message}")

def log_error(message):
    """Affiche une erreur"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ❌ {message}")

def log_section(title):
    """Affiche un séparateur de section"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

if __name__ == "__main__":
    log_section("TEST DU LOGGER")
    log_info("Ceci est une information")
    log_success("Opération réussie !")
    log_warning("Attention, quelque chose d'étrange")
    log_error("Une erreur s'est produite")
    log_section("FIN DES TESTS")
