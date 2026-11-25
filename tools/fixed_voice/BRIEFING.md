# Briefing : contexte Omni-FR et choix de modèles pour la voix fixe

Rédigé par Claude (session de travail sur la branche `omni-fr`) pour l'agent qui développe
`tools/fixed_voice/`. Recherches web faites le 2026-11-20 et le 2026-11-25 ; les chiffres
marqués *estimé* n'ont pas été mesurés.

## 1. D'où vient le besoin

Jarvis tourne nativement sous **Windows**, sur une **RTX 2070 8 Go** (448 Go/s, Turing sm75),
sur la branche `omni-fr` (dernier commit `ce1181f`).

**Pile en service :**
- **LLM :** Nemotron-3.5-Bolt-E4B UD-Q4_K_XL (base Gemma 4) via llama-server, sur le GPU.
- **Reconnaissance vocale :** Parakeet TDT 0.6B v3 via `onnx-asr`, sur le processeur.
- **TTS :** Qwen3-TTS 0.6B Base avec clonage de voix (`faster-qwen3-tts`, graphes CUDA, environnement séparé), ou Kokoro-82M (`kokoro-onnx`, voix `ff_siwis`).

Mesures relevées dans les logs de l'utilisateur (`JARVIS_TIMINGS=1`) :

| | Qwen3-TTS (clone) | Kokoro |
|---|---|---|
| Début de la voix après le texte | 1,6 à 1,7 s (4,0 s au premier appel avant le warm-up) | 0,6 à 1,1 s |
| VRAM totale de la machine | ~7,1 à 7,3 Go sur 8 | ~5,1 Go |

Qwen3 génère l'audio pas à pas, à peine plus vite que le temps réel sur la 2070. Jarvis doit
donc accumuler ~1,3 s d'avance avant de jouer. Le but de la voix fixe est triple :
- descendre le début de la voix vers ~0,1 à 0,3 s (*estimé*) ;
- libérer ~2 Go de VRAM, ce qui permettrait de passer la reconnaissance vocale sur le GPU ;
- rendre un Mac envisageable : Qwen3 accéléré dépend de CUDA, Piper tourne sur processeur.

**Contraintes :**
- usage personnel non commercial ;
- clonage uniquement avec le consentement de la personne dont c'est la voix ;
- français d'abord ;
- tout en local.

## 2. Remarques sur la branche `fixed-voice-distillation`

**Base de la branche.** Elle part de `main` (`9d90257`), pas de `omni-fr`. `tools/fixed_voice/`
est autonome, donc rien ne casse. Mais l'intégration dans Jarvis se fera sur `omni-fr` : le
moteur `piper` y existe déjà (`tts_engine: "piper"`, `tts_piper_model_path`,
`tts_piper_length_scale`, …). Un rebase sur `omni-fr` est à prévoir avant la fusion.

**Revue de `distill.py`, par ordre d'impact :**

1. **Ne pas partir de zéro.** Le README dit « Training starts from scratch ». Partir d'un
   checkpoint Piper *medium* français divise fortement les besoins en données et en temps.
   La doc du trainer maintenu juge ça « highly recommended […] even if the checkpoint is from
   a different language ». Les checkpoints français disponibles sont `fr_FR/siwis`,
   `fr_FR/upmc` et `fr_FR/mls` dans `rhasspy/piper-checkpoints`.
2. **Utiliser le trainer maintenu.** `rhasspy/piper` est archivé, et son successeur est
   `OHF-Voice/piper1-gpl` :
   - commande : `python3 -m piper.train fit --data.voice_name … --data.csv_path …
     --data.audio_dir … --data.espeak_voice fr-fr --model.sample_rate 22050
     --data.cache_dir … --data.config_path … --data.batch_size 32 --ckpt_path <medium.ckpt>` ;
   - format du CSV : `fichier.wav|texte` (avec l'extension), et non `id|texte` comme dans
     LJSpeech ;
   - sans réglage supplémentaire, seuls les checkpoints *medium* sont pris en charge.
3. **Filtrer ce que génère le modèle source.** Qwen3 saute, répète ou invente parfois des mots,
   et le modèle élève apprendrait ces erreurs. Il faut un aller-retour par la reconnaissance
   vocale : Parakeet TDT 0.6B v3 via `onnx-asr`, déjà une dépendance d'`omni-fr`. On rejette au-delà
   d'un seuil d'erreur sur les caractères, et on écarte aussi les débits aberrants (secondes
   par caractère). tiny-voca et l'exemple Hackaday font cette vérification.
4. **Ne pas tout arrêter sur une erreur.** `generate` lève une exception au premier audio
   invalide et interrompt toute la génération. Mieux vaut journaliser, sauter la phrase et
   continuer.
5. **Taille du modèle source.** Pour la génération hors ligne, Jarvis arrêté, la 1.7B-Base
   devrait mieux cloner que la 0.6B ; tiny-voca a utilisé la 1.7B Base. Qu'elle tienne en fp16
   dans 8 Go reste *à vérifier*.
6. **Vitesse de génération.** Le paquet officiel `qwen-tts` est lent. L'environnement Qwen3
   d'`omni-fr` a déjà `faster-qwen3-tts` (graphes CUDA), qui va à peu près au temps réel sur la
   2070. Ordre de grandeur : 2 à 3 h d'audio, soit environ une nuit.
7. **Extrait de référence.** Le nettoyer une fois avec `inworld/voice-isolator` (Apache-2.0,
   48 kHz) s'il est bruité. Il n'enlève que le bruit de fond, pas les autres voix.
8. **Corpus.** Le texte doit avoir la même forme que ce que Jarvis prononcera : chiffres ou
   lettres, heures, unités, noms propres, questions, phrases très courtes (« C'est fait. »).
   Il faut aussi des phrases de domotique, et un jeu de phrases tenu à l'écart pour les écoutes
   comparatives entre le modèle source et le modèle élève.
9. **Windows.** L'entraînement Piper demande `build-essential`, `cmake` et la compilation de
   `monotonic_align` : en pratique sous Linux, donc WSL2 sur la 2070. La génération peut tourner
   nativement sous Windows.

## 3. Panorama des modèles (vérifié en ligne)

### Modèles sources possibles

| Modèle | Taille | Français | Clonage | Licence | Verdict |
|---|---|---|---|---|---|
| **Reverb-TTS** (`software-mansion/Reverb-TTS`) | 297 M (319 M dans un tableau) | oui (7 langues) | oui, sans entraînement, avec 3 à 10 s de référence | code Apache-2.0, **poids CC BY-NC 4.0** | **à tester en premier** (voir §4) |
| Qwen3-TTS 1.7B / 0.6B Base | 1,7 B / 0,6 B | oui | oui | Apache-2.0, aucune restriction sur l'usage des sorties | modèle source par défaut |
| Ming-3.0-tts-0.5B (inclusionAI) | 0,5 B (1 B selon la page) | oui | oui | MIT | non : la fiche dit « French SIM is the clearest multilingual gap », pas de chiffres de latence, testé seulement sur H800/H20 |
| OmniVoice (base de Reverb ; portage GGUF `Serveurperso/OmniVoice-GGUF`) | 0,6 B | à vérifier | oui | à vérifier | Reverb en est la version allégée et retravaillée |

Reverb-TTS en détail :
- **Architecture :** OmniVoice (0,6 B, diffusion masquée), élagué, distillé, réentraîné sur 7 langues, puis affiné par DPO itératif.
- **Vitesse annoncée :** « over 20× » le temps réel sur Apple M4 Pro et iPhone 17 en batch 1, et 232× sur H200 en batch 32.
- **Formats :** export ExecuTorch (MLX, Vulkan, QNN), PyTorch en fp32/fp16/int8 (par défaut)/int4.
- **Non précisé :** pas d'ONNX, ni streaming, ni processeur mentionnés.
- **Rien n'est mesuré sur une 2070.**

### Modèles élèves possibles

| Modèle | Taille | Vitesse | État | Remarque |
|---|---|---|---|---|
| **Piper VITS medium** | ~20 à 60 M | très rapide sur processeur | trainer maintenu (piper1-gpl), checkpoints français | choix sûr, à affiner depuis un checkpoint français |
| **tiny-voca** (`n4ze3m/tiny-voca`) | 4,5 M | ~70× le temps réel sur 4 threads de processeur | Apache-2.0, code d'entraînement, export ONNX | déjà distillé depuis Qwen3-TTS (17,6 h, 2 voix, 7,7 h d'entraînement sur une 4090). **Anglais seulement** : convertisseur en phonèmes Dizione, et durées étiquetées par wav2vec2. Le porter en français demande espeak-ng `fr-fr` et un aligneur de phonèmes multilingue. Son un peu synthétique (synthétiseur source-filtre). |
| Kokoro-82M / StyleTTS2 | 82 M | rapide | outils d'entraînement communautaires, non vérifiés | sert déjà de point de comparaison à l'exécution (`ff_siwis`, noté B-) |

### Outils annexes

- `inworld/voice-isolator` : nettoyage de la référence et curation du corpus. Ce n'est pas un séparateur de voix.
- Parakeet TDT 0.6B v3 : filtrage par aller-retour en reconnaissance vocale.
- wav2vec2, phonèmes multilingues : uniquement pour un portage de tiny-voca.

## 4. Ordre recommandé

0. **Tester Reverb-TTS sur la 2070 avant tout entraînement.** À mesurer : le temps avant le
   premier son pour une phrase de 2 à 3 s, la VRAM en int8, et la ressemblance de la voix
   française clonée.
   - S'il tient environ 10× le temps réel ou plus sur la 2070, il pourrait remplacer Qwen3
     directement à l'exécution, sans distillation.
   - Sinon, il peut au moins servir de modèle source plus rapide.
   - Ses poids sont NC, ce qui convient à cet usage personnel non commercial. Un modèle élève
     entraîné sur ses sorties doit rester dans ce cadre.
1. Nettoyer la référence (voice-isolator) et préparer le corpus (§2, point 8).
2. Générer 2 à 3 h d'audio, puis filtrer avec Parakeet.
3. Affiner Piper medium depuis `fr_FR/siwis` ou `fr_FR/upmc` avec piper1-gpl, sous WSL2.
4. Faire des écoutes à l'aveugle, sur le jeu de phrases tenu à l'écart, entre le modèle source et le modèle élève.
5. Brancher le résultat dans `omni-fr` via `tts_engine: "piper"`, et mesurer la ligne
   `⏱️ heard → voice`.
6. Plus tard, éventuellement : porter tiny-voca en français, avec 15 à 20 h d'audio. Il serait plus léger encore, utilisable sur Mac M1 ou dans un navigateur.

## Sources

- Reverb-TTS : https://huggingface.co/software-mansion/Reverb-TTS
- Qwen3-TTS 1.7B Base (licence) : https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- tiny-voca : https://huggingface.co/n4ze3m/tiny-voca · https://github.com/n4ze3m/tiny-voca
- Ming-3.0-tts-0.5B : https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B
- voice-isolator : https://huggingface.co/inworld/voice-isolator
- Entraînement piper1-gpl : https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md
- Checkpoints Piper : https://huggingface.co/datasets/rhasspy/piper-checkpoints
- Méthode « voix clonée → Piper » : https://hackaday.com/2025/07/09/how-to-train-a-new-voice-for-piper-with-only-a-single-phrase
