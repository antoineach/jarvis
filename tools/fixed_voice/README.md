# Train a fixed Jarvis voice

Qwen3-TTS reads your sentences with one reference voice. A single-speaker Piper VITS student learns from those text/audio pairs. Its ONNX graph includes waveform generation. Deployment needs Piper and two voice files. Training uses local files and keeps the reference recording on your machine.

## Generate speech

In a Python 3.12 environment install the teacher with `pip install qwen-tts soundfile`. Download the full Qwen/Qwen3-TTS-12Hz-0.6B-Base model directory before going offline. Use a clear recording of the target voice, its exact transcript, and a UTF-8 text file with one sentence per line in your chosen language. Include the names, numbers, questions and short replies Jarvis will actually say. Reserve some sentences separately for listening tests.

From this directory run:

```sh
python distill.py generate --teacher /models/Qwen3-TTS-12Hz-0.6B-Base \
  --reference reference.wav --transcript reference.txt --language French \
  --sentences french.txt --dataset data/fr
```

French is an example. Choose one language per dataset. Rerun the same command after interruption to reuse completed WAV files. Run one generator at a time per directory. Keep the teacher directory immutable. Listen to samples and remove misread sentences from the input before regenerating the manifest.

## Fit and export the student

In a separate GPU environment install the archived rhasspy/piper trainer, including its compiled monotonic alignment extension, following the upstream installation steps. This script uses the legacy `piper_train.preprocess` and Lightning CLI. [Piper training instructions](https://github.com/rhasspy/piper/blob/master/TRAINING.md)

```sh
python distill.py train --dataset data/fr --work runs/fr --batch-size 16 --epochs 2000
find runs/fr -name '*.ckpt'
python distill.py export --work runs/fr --checkpoint /path/to/chosen.ckpt --output voices/fr
echo 'Bonjour, comment puis-je vous aider ?' | piper -m voices/fr/voice.onnx --output_file sample.wav
```

Use a new work directory for each training run. Training starts from scratch. A few sentences can exercise the pipeline, but learning pronunciation and speaker identity requires a broad corpus. Compare held-out sentences from the teacher and student before choosing a checkpoint. Export checks the inference graph against a strict 100 million weight-element limit. The training discriminator and teacher stay on the training machine.
