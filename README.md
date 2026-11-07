# LightShift-PatchCore

Projet de détection d'anomalies industrielles robuste aux changements d'éclairage.

## Objectif
Comparer PatchCore vanilla à des variantes avec :
- B1 : banque mémoire enrichie par augmentations photométriques ;
- B2 : alignement statistique des caractéristiques ;
- B3 : recalibration conforme du seuil ;
- B4 : adaptateur appris (optionnel).

Le périmètre initial est volontairement limité : reproduire PatchCore sur trois catégories MVTec AD, générer des décalages photométriques, puis implémenter B2/B3 et B1.

## Démarrage rapide — Windows PowerShell

Prérequis : Python 3.11 x64, Git et `uv`.

```powershell
# 1. Vérifier les outils
py --version
git --version

# 2. Créer l'environnement
uv venv --python 3.11
.venv\Scripts\Activate.ps1

# 3. Installer les dépendances du projet
uv sync --extra dev

# 4. Vérifier l'environnement
uv run python scripts/check_environment.py
uv run pytest
```

### PyTorch : choisir CPU ou CUDA
Commencez en CPU si vous ne connaissez pas encore votre GPU. Pour CUDA, identifiez d'abord le modèle NVIDIA et la version du pilote avec `nvidia-smi`, puis utilisez le sélecteur officiel PyTorch :
https://pytorch.org/get-started/locally/

Si vous choisissez l'installation CPU, installez le build CPU officiel avant les autres paquets :

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Ensuite :
```powershell
uv sync --extra dev
```

Si `uv sync` remplace votre build PyTorch CPU par un autre build, utilisez le sélecteur PyTorch et vérifiez `torch.cuda.is_available()` après installation. Ne mélangez pas des builds CPU/CUDA au hasard.

### Anomalib
Anomalib est inclus dans les dépendances. Après installation, vérifiez :
```powershell
uv run python -c "import anomalib; print('anomalib OK')"
```
La version exacte de l'API Anomalib doit être figée après le premier test de reproduction. Le code du projet gardera l'accès à Anomalib isolé dans `src/lightshift/models/`.

### FAISS
FAISS est volontairement optionnel au démarrage. Sur Windows, la documentation officielle recommande Conda pour l'installation des paquets précompilés et indique que `faiss-cpu` est disponible pour Windows x86-64. N'installez pas FAISS GPU sous Windows : les paquets GPU documentés sont Linux x86-64.

Option Windows CPU (dans un environnement Conda/Miniforge séparé ou intégré avec précaution) :
```powershell
conda install -c pytorch -c conda-forge faiss-cpu
```

Sur Colab/Kaggle Linux, essayez `uv pip install faiss-cpu` ou la méthode Conda documentée. Si l'installation échoue, utilisez provisoirement `sklearn.neighbors` ou `torch.cdist` pour le baseline et revenez à FAISS ensuite.

## Exécution locale
```powershell
uv run python scripts/check_environment.py
uv run pytest
uv run streamlit run app/Home.py
```

L'application et les modules ML sont des squelettes : les fonctions métier seront implémentées après validation du baseline.

## Google Colab
1. Ouvrir un notebook Colab et activer un runtime GPU : Runtime > Change runtime type > T4 GPU (si disponible).
2. Cloner le dépôt :
```bash
!git clone <URL_DU_DEPOT_GITHUB>
%cd LightShift-PatchCore
!python -m pip install --upgrade pip
!pip install uv
!uv sync --extra dev
!uv run python scripts/check_environment.py
```
3. Vérifier que `torch.cuda.is_available()` retourne `True`.
4. Les données et résultats doivent être stockés dans Google Drive ou téléchargés vers le dépôt/stockage d'équipe. Ne commitez jamais les images MVTec dans Git.

Pour Kaggle : créer un Notebook, activer l'accélérateur GPU dans Settings, cloner le dépôt et installer les dépendances de la même manière.

## Structure
```text
configs/       configurations YAML
data/          données locales (non versionnées)
splits/        listes d'identifiants de données figées
src/           code Python du projet
app/           application Streamlit
scripts/       scripts utilitaires
tests/         tests unitaires et anti-fuite
notebooks/     exploration et analyse
results/       métriques, figures et tableaux (selon politique de versionnement)
paper/         article LaTeX
presentation/  diapositives
```

## Règles scientifiques à respecter
- Séparer train, calibration B2/B3, validation et test.
- Ne jamais utiliser les images de test pour estimer B2 ou calibrer B3.
- Ne jamais régler les hyperparamètres sur le test.
- Comparer B1 à budget mémoire/coreset égal.
- Rapporter toutes les catégories, y compris les échecs.
- Enregistrer seed, configuration, commit Git et versions des bibliothèques pour chaque expérience.

## Prochain jalon
Obtenir un premier run PatchCore vanilla sur trois catégories, puis comparer les résultats à la référence publiée avant de développer B1/B2/B3.
