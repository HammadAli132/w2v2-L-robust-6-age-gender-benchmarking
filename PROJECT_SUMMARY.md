# Project Summary: Speech-Based Age and Gender Classification

## Overview

This project is an **inference/demo project**, not a model-training project. It uses a pretrained, fine-tuned wav2vec 2.0 model to estimate a speaker's age and classify the voice as **female**, **male**, or **child** from an audio recording.

The repository contains only three original artifacts:

| Item | Purpose | Size |
| --- | --- | ---: |
| `notebookbef6cbf75b.ipynb` | End-to-end Jupyter/Kaggle inference demo | 101,017 bytes |
| `model/model.yaml` | `audonnx` model configuration and output labels | 162 bytes |
| `model/model.onnx` | Serialized neural-network weights and graph | 363,421,332 bytes (~347 MiB) |

## Model

The model is **`w2v2-L-robust-6-age-gender`**, referenced from [Zenodo record 7761387](https://doi.org/10.5281/zenodo.7761387). It is a wav2vec 2.0 speech representation model fine-tuned for simultaneous age regression and gender-category classification.

The ONNX graph contains:

- Seven convolutional wav2vec2 feature-extractor blocks (`conv_layers.0` through `conv_layers.6`).
- Six transformer encoder layers (`encoder.layers.0` through `encoder.layers.5`).
- A shared `hidden_states` representation.
- A separate age head with dense, `tanh`, and output-projection operations.
- A separate gender head with dense, `tanh`, and output-projection operations.

The model's exposed interface, as reported by the executed notebook, is:

- Input: `signal`, a floating-point waveform with shape `[1, variable_length]`.
- Output: `hidden_states`.
- Output: `logits_age`, containing one normalized age estimate.
- Output: `logits_gender`, containing three scores.

`model.yaml` declares that the age output is labeled `age`, while the gender output positions are ordered as `female`, `male`, and `child`. It applies no configured input or output transform (`transform: null`) and targets `audonnx` model format version 0.6.3.

The notebook interprets the outputs as follows:

```text
age in years = logits_age[0][0] * 100
gender       = argmax(logits_gender[0])
```

The gender outputs are displayed as **raw logit scores**, not normalized probabilities. The notebook does not use the exposed `hidden_states` output.

## Notebook Workflow

The notebook has 20 cells: 11 Markdown cells and 9 executed code cells. Its flow is:

1. Install `audonnx`, `audeer`, `audinterface`, and `ipywidgets`.
2. Download the model archive from Zenodo into `cache/model.zip` if it is absent.
3. Extract the archive into `model/` if that directory is absent.
4. Load the local model with `audonnx.load('model')`.
5. Define audio loading, prediction, and visualization helpers.
6. Run a one-second random-noise smoke test.
7. Provide an `ipywidgets` uploader for one or more audio files.
8. Provide browser-side JavaScript for microphone recording.
9. Support classifying a single audio file by path.
10. Support classifying the audio files immediately inside one directory.
11. Plot aggregate age and gender distributions for batch results.

The stored notebook output shows that all nine code cells ran. The model loaded successfully and reported the interface above. The random-noise smoke test returned `40.2 years` and `female`, but the notebook correctly notes that a prediction on noise is meaningless. The path example did not run because `your_audio.wav` was a placeholder, and the batch example found no audio files in `/kaggle/input`.

## Audio and Inference Pipeline

All accepted audio is converted through the system `ffmpeg` executable to:

- 16 kHz sampling rate;
- mono audio;
- temporary WAV format.

The WAV PCM samples are read as signed 16-bit integers, converted to `float32`, and scaled to approximately `[-1, 1]`. The resulting one-dimensional array is passed to the `audonnx` wrapper together with the 16 kHz sampling rate.

For each prediction, the notebook:

- multiplies the normalized age output by 100 and rounds it to one decimal place;
- selects the largest of the three gender logits;
- plots age on a 0–100 horizontal gauge;
- plots the three raw gender logits and outlines the winning bar;
- optionally embeds an audio player for the processed signal.

Batch inference builds a pandas table with the file name, predicted age, selected gender, and all three raw gender scores. It can then plot an age histogram and counts for each selected gender category.

## Ways to Supply Audio

The notebook supports four usage paths:

- Upload one or more audio files through an `ipywidgets.FileUpload` control.
- Record audio with the browser microphone and pass base64-encoded WebM data back to the Python kernel.
- Set `AUDIO_PATH` to one local file.
- Set `AUDIO_DIR` to a directory containing supported files.

Recognized batch extensions are `.wav`, `.mp3`, `.ogg`, `.flac`, `.webm`, and `.m4a`. Batch discovery is not recursive.

## Runtime Requirements

The effective requirements are:

- Python/Jupyter environment;
- `audonnx`, `audeer`, `audinterface`, `numpy`, `pandas`, `matplotlib`, and `ipywidgets`;
- a system installation of `ffmpeg`;
- browser microphone permission for recording;
- internet access only when the model archive or Python dependencies are not already available.

The notebook was saved with Python 3.12.13 metadata and appears designed primarily for Kaggle or a classic Jupyter environment. Its recorded Papermill execution lasted about 160 seconds on 2026-09-23.

## Important Limitations and Risks

- The model estimates traits from voice and can be inaccurate or biased across accents, languages, recording conditions, and demographic groups. Its results should not be treated as verified identity or used for consequential decisions.
- The model's stated license is **CC BY-NC-SA 4.0**, so it is not licensed here for commercial use, and attribution/share-alike obligations apply.
- `female`, `male`, and `child` are the model's three acoustic classes. “Child” is treated as mutually exclusive with female/male, so the output is not a general-purpose gender representation.
- The displayed gender scores are logits. They should not be described as confidence percentages unless an appropriate calibration/normalization step is added and validated.
- Age is hard-scaled by 100 based on the model convention. The visualization fixes its axis at 0–100, but the code does not clamp out-of-range predictions.
- Audio conversion does not check the `ffmpeg` process return code before attempting to open the output WAV, so conversion failures may surface as less-informative WAV-reading errors.
- Temporary paths are created with `tempfile.mktemp()`, which is weaker and more collision-prone than securely creating a named temporary file.
- The microphone JavaScript relies on classic notebook globals such as `Jupyter.notebook` or `IPython.notebook`. It may not work unchanged in JupyterLab, modern notebook front ends, or Kaggle environments that isolate output JavaScript.
- The batch loader only scans files directly inside `AUDIO_DIR`; it does not walk subdirectories.
- The notebook downloads only when `cache/model.zip` is missing and extracts only when the entire `model/` directory is missing. A partial or corrupt existing download/directory is therefore not repaired automatically.
- No formal evaluation data, accuracy metrics, calibration analysis, automated tests, or training code are included in this directory.

## Provenance

The notebook cites:

> Felix Burkhardt, Johannes Wagner, Hagen Wierstorf, Florian Eyben, and Björn Schuller. “Speech-based Age and Gender Prediction with Transformers.” 15th ITG Conference on Speech Communication, 2023.

It also points to the upstream example code at [audeering/w2v2-age-gender-how-to](https://github.com/audeering/w2v2-age-gender-how-to).

## Bottom Line

This directory packages a large, ready-to-run ONNX speech model with an interactive demonstration notebook. The notebook handles common audio inputs, normalizes them to the model's expected 16 kHz mono waveform, runs joint age/gender inference, and visualizes individual or batch results. It is suitable as a prototype or model-testing harness, subject to its non-commercial license, environment-specific UI code, and the accuracy, fairness, and interpretation limitations described above.
