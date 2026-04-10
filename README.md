# Convolutional Neural Networks — course materials

This repository supports a CNN track using **Keras 3** and **TensorFlow**: an MNIST lab, topic notebooks, and the **`CourseNotebooks`** sequence (including transfer learning on dogs vs cats).

---

## Environment

Use a virtual environment (`.venv` is gitignored):

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is **pinned** (`pip freeze`) for reproducible installs. For Jupyter, you can register a kernel from this env, e.g. `python -m ipykernel install --user --name=convnet-course`.

---

## Dataset for dogs/cats notebooks (`05A`–`05D`)

Notebooks expect **`data/kaggle_dogs_vs_cats_small/`** with `train/`, `validation/`, and `test/` (each split has `cat/` and `dog/` class folders).

- **Quick setup (no Kaggle account):** from the **repo root**, run:

  ```bash
  python scripts/bootstrap_dogs_vs_cats_small.py
  ```

  This uses TensorFlow Datasets **`cats_vs_dogs`** (first run downloads ~786 MB) and writes the same counts as the course recipe (e.g. 2000 train, 1000 val, 2000 test images total).

- **Course path:** use **`CourseNotebooks/05A_asirra_the_dogs_vs_cats_dataset.ipynb`** after you have the full Kaggle train set extracted under `data/kaggle_dogs_vs_cats/` as described there.

The `05D` notebook resolves `data/` whether your Jupyter **working directory** is the repo root or `CourseNotebooks/`.

---

## `CourseNotebooks/05D_fine_tuning_vgg16.ipynb` — transfer learning & fine-tuning

Adapted from Chollet’s [Deep Learning with Python (Ch. 8) notebook](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter08_intro-to-dl-for-computer-vision.ipynb) (Apache-2.0). See also the [Keras transfer learning guide](https://keras.io/guides/transfer_learning/).

### What you do in this notebook

1. **Feature extraction with ImageNet VGG16** (`include_top=False`, fixed `input_shape` e.g. 180×180×3).
2. **Fast path:** precompute VGG16 features, train a small dense head on top (no augmentation in that variant).
3. **Stronger path:** freeze VGG16, add **data augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`), **`vgg16.preprocess_input`**, then train a custom classifier end-to-end on image batches.
4. **Fine-tuning:** unfreeze the top convolutional layers of VGG16, recompile with a **small learning rate** (e.g. RMSprop `1e-5`), and continue training with `ModelCheckpoint` on `val_loss`.
5. **Bonus — ResNet50:** same pipeline on the **same split**, using `keras.applications.resnet50.ResNet50` and **`resnet50.preprocess_input`**, with wall-clock timing for head training + fine-tune and final **test accuracy** (compare to VGG16).
6. **Challenge section:** objectives, tasks, reflection questions, and **solutions** tied to the notebook’s saved metrics when you run the cells.

### Practical notes

- Training VGG/ResNet with augmentation and fine-tuning is **slow on CPU** (on the order of hours for full epoch counts); use a GPU or reduce epochs for smoke tests (`EPOCHS_HEAD` / `EPOCHS_FT` in the ResNet cell).
- Saved models go under **`./models/`** (e.g. `feature_extraction.keras`, `fine_tuning.keras`, ResNet checkpoints). Create `models` if needed.
- The ResNet bonus cell includes **its own imports** so it can be run after a restart if the dataset path resolves correctly.

---

## MNIST CNN lab (`ConvolutionalNeuralNetworks.ipynb`)

**Learning objectives**

- CNN architecture and training with Keras / TensorFlow.
- Image preprocessing for deep learning.
- Compare CNN vs dense networks on **MNIST**.
- Evaluation, prediction, and saving models in modern Keras.

**In the notebook you:** explore MNIST, build and train a CNN, compare to a dense baseline, practice predictions, and export models.

---

## Suggested path through the repo

1. Work through the topics in **`ConvolutionalNeuralNetworks.ipynb`** (TOC and demos).
2. Run the notebooks in **`CourseNotebooks/`**, answer the challenges, and experiment with architectures and hyperparameters.
3. For **`05D`**, prepare **`data/kaggle_dogs_vs_cats_small`** (bootstrap script or `05A`), then run through feature extraction → augmentation → fine-tuning → optional ResNet50 comparison.

---

Happy learning. For API details, use the [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) docs or ask your instructor.
