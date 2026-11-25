# Une voix française fixe pour Jarvis

L’Ada 32 Go sert à fabriquer le corpus avec Qwen3-TTS Base 1.7B et à affiner Piper VITS medium depuis SIWIS. Le Mac mini M4 utilise seulement Piper sur CPU, le fichier ONNX et sa configuration. Le LLM peut continuer à utiliser Metal. Le nombre de paramètres, le poids du fichier, la mémoire du processus et le délai avant le premier bloc audio sont des mesures distinctes.

## 1. Installer sur l’Ada (Linux ou WSL2)

Depuis ce dossier, dans Bash, avec Python 3.12, un pilote NVIDIA fonctionnel et suffisamment de place pour les modèles, le cache audio et plusieurs checkpoints. Les environnements séparés évitent de faire dépendre Qwen du jeu de dépendances de Lightning.

```bash
sudo apt-get install build-essential cmake ninja-build python3-dev libsndfile1 sox ffmpeg
python3.12 -m venv .teacher
.teacher/bin/pip install qwen-tts soundfile 'onnx-asr[cpu,hub]'
.teacher/bin/python -c 'import torch; print(torch.cuda.get_device_name(0))'
.teacher/bin/huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/qwen
.teacher/bin/huggingface-cli download istupakov/parakeet-tdt-0.6b-v3-onnx --include "*.int8.onnx" "nemo128.onnx" "vocab.txt" "config.json" --local-dir models/parakeet
.teacher/bin/huggingface-cli download --repo-type dataset rhasspy/piper-checkpoints 'fr/fr_FR/siwis/medium/epoch=815-step=671824.ckpt' --local-dir models/base
git clone https://github.com/OHF-Voice/piper1-gpl vendor/piper
git -C vendor/piper checkout 6aacdbc98eda4b1cb81d960bef7c7fae710b13fd
python3.12 -m venv .trainer
source .trainer/bin/activate
(cd vendor/piper && pip install -e '.[train]' && ./build_monotonic_align.sh && python setup.py build_ext --inplace)
python -c 'import torch; print(torch.cuda.get_device_name(0))'
python -m piper.train fit --help
deactivate
```

Si `huggingface-cli` est remplacé par `hf` dans ta version du Hub, utilise `.teacher/bin/hf download` avec les mêmes arguments. Si PyTorch ne détecte pas la carte, installer sa distribution CUDA compatible avec le pilote dans l’environnement concerné avant la suite. Les téléchargements sont une étape de préparation. Les fichiers de voix et les textes restent locaux.

## 2. Valider la voix puis fabriquer le corpus

Préparer un extrait propre de la voix cible, une seule personne, sans musique ni réverbération marquée, et sa transcription exacte en UTF-8. Comparer d’abord les seize phrases pilotes à l’extrait réel. Écouter le timbre, les fins de phrases, l’accent et les questions. Un mauvais clone au stade professeur justifie de changer d’extrait avant de générer plusieurs heures. Utiliser un nouveau dossier de données si la référence change.

```bash
.teacher/bin/python distill.py generate --teacher models/qwen --reference reference.wav --transcript reference.txt --language French --sentences pilot-fr.txt --dataset data/fr
curl -fL https://downloads.tatoeba.org/exports/per_language/fra/fra_sentences.tsv.bz2 -o fra.tsv.bz2
python3 corpus.py fra.tsv.bz2 train-fr.txt
.teacher/bin/python distill.py generate --teacher models/qwen --reference reference.wav --transcript reference.txt --language French --sentences train-fr.txt --dataset data/fr
.teacher/bin/python curate.py --dataset data/fr --asr models/parakeet --heldout heldout-fr.txt
```

Le corpus de départ comprend 3 200 phrases françaises distinctes de Tatoeba et les répliques du pilote. Pour une voix destinée à Jarvis, passer ses propres réponses représentatives à `corpus.py --domain`, notamment noms propres, confirmations, météo, horaires et nombres écrits comme ils se prononcent. Le fichier compagnon conserve les identifiants des phrases. Conserver la provenance et consulter les conditions de réutilisation avant redistribution. [Tatoeba](https://tatoeba.org/fr/terms_of_use)

La génération réutilise les WAV terminés. Lancer un seul générateur par dossier. La curation exclut le texte du jeu d’écoute, les durées hors plage et les phrases dont le taux d’erreur sur caractères normalisés dépasse 12 %. Examiner `review.jsonl` et écouter aussi des exemples acceptés. L’ASR peut hésiter sur les noms et les nombres, et ne juge ni le timbre ni la prosodie. La durée acceptée affichée est la vraie taille du corpus. Au besoin, corriger le texte ou supprimer le WAV concerné puis relancer génération et curation.

## 3. Affiner et exporter

```bash
base='models/base/fr/fr_FR/siwis/medium/epoch=815-step=671824.ckpt'
.trainer/bin/python distill.py train --base "$base" --dataset data/fr --work runs/fr --batch-size 32 --epochs 2000
find runs/fr -name '*.ckpt'
.trainer/bin/python distill.py export --work runs/fr --checkpoint /chemin/du/checkpoint-choisi.ckpt --output voices/essai-1
```

Le warm-start charge les poids et démarre un nouvel optimiseur. Une reprise après interruption utilise les mêmes données, paramètres et dossier avec `--resume /chemin/last.ckpt` à la place de `--base`. Le batch 32 est un point de départ pour 32 Go. Descendre à 16 si les phrases longues dépassent la mémoire. Pour budgéter le calcul, chronométrer quelques époques après la préparation du cache. Multiplier leur durée médiane par le nombre d’époques restantes. Les 2 000 époques sont le point de départ documenté par Piper pour un petit corpus. Exporter plusieurs candidats dans des dossiers distincts et les écouter en cours de route. [Recette amont](https://github.com/OHF-Voice/piper1-gpl/blob/6aacdbc98eda4b1cb81d960bef7c7fae710b13fd/docs/TRAINING.md)

## 4. Mesurer et déployer sur le Mac mini M4

Copier ce dossier de scripts, `voices/essai-1/voice.onnx` et `voice.onnx.json` sur le Mac. Utiliser un Python 3.12 **arm64** natif. Pour obtenir un premier chiffre avant même l’entraînement, télécharger la voix SIWIS publique. Après installation, la synthèse fonctionne localement.

```bash
python3.12 -m venv .mac
.mac/bin/pip install piper-tts==1.3.0
.mac/bin/python -c 'import platform; print(platform.machine())'
.mac/bin/python -m piper.download_voices fr_FR-siwis-medium
.mac/bin/python m4_bench.py fr_FR-siwis-medium.onnx heldout-fr.txt resultats-siwis
.mac/bin/python m4_bench.py voices/essai-1/voice.onnx heldout-fr.txt resultats-essai-1
afplay resultats-essai-1/004.wav
```

Chaque essai produit les WAV et un `metrics.json` avec temps de chargement, premier bloc PCM à froid et p50/p95 à chaud, facteur temps réel (plus petit est mieux), taille ONNX et pic mémoire du processus. Le premier bloc correspond généralement à une phrase entière. Le délai matériel de lecture, le LLM et l’ASR sont exclus de cette mesure. Répéter avec Jarvis et son LLM chargés. Dans la configuration de la branche `omni-fr`, choisir `tts_engine: piper` et définir `tts_piper_model_path` sur le chemin absolu du ONNX. Conserver le JSON adjacent. Mesurer aussi le délai de bout en bout avec `JARVIS_TIMINGS=1`.

Pour sélectionner le checkpoint, garder les mêmes phrases inédites et le même volume d’écoute. Comparer les WAV dans un ordre masqué avec ceux du professeur (générer `heldout-fr.txt` dans un dossier distinct) et écouter la référence réelle. Relever séparément mots incorrects, ressemblance et naturel. Départager les candidats de qualité proche par le p95 et la mémoire mesurés sur le M4. Un objectif de premier bloc à chaud sous 300 ms sur les répliques courtes est une cible à tester. Le code impose moins de 100 millions d’éléments de poids à l’export. Aucune mesure matérielle ni supériorité sur les autres voix du marché n’est présumée.

Pour préparer l’écoute à l’aveugle, lancer `python3 blind.py ecoute resultats-siwis resultats-essai-1` (ajouter autant de dossiers candidats que nécessaire). Remplir `ecoute/scores.csv` puis ouvrir `ecoute/key.csv` pour retrouver les modèles.
