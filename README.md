# Convolutional Neural Networks — course materials

This repository supports a CNN track using **Keras 3** and **TensorFlow**: an MNIST lab, topic notebooks, and the **`CourseNotebooks`** sequence (including transfer learning on dogs vs cats).

## Group members

- Preeja Anilal(8791796)
- Izevbokun(9016626)
- Minh Thuan (8730956)

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

The notebook walks **ImageNet-pretrained VGG16** on the **dogs vs cats small** split: it resolves `data/kaggle_dogs_vs_cats_small` from either the repo root or `CourseNotebooks/` (same logic as the first data-loading cell). Short **“To the Student”** prompts explain `include_top=False`, `input_shape`, and how conv vs pool blocks are structured in VGG16.

### What you do in this notebook

1. **Instantiate the VGG16 convolutional base** (`weights="imagenet"`, `include_top=False`, `input_shape=(180, 180, 3)`).
2. **Fast feature extraction (no augmentation):** run the frozen base on the datasets, **precompute** bottleneck features and labels, then train a small **dense classifier** for **20 epochs** with `ModelCheckpoint` on `val_loss`.
3. **Feature extraction with augmentation:** keep VGG16 frozen, build an input pipeline with **`RandomFlip`**, **`RandomRotation`**, **`RandomZoom`**, and **`keras.applications.vgg16.preprocess_input`**, then train the head for **50 epochs** (again checkpointing on `val_loss`). Plot train/val accuracy and loss.
4. **Fine-tuning:** unfreeze only the **last four layers** of the conv base (`conv_base.layers[:-4]` remain frozen), **recompile** with **RMSprop, learning rate `1e-5`**, train **30 epochs** with checkpoint **`./models/fine_tuning.keras`**, then evaluate on the test set and plot curves.
5. **Bonus — ResNet50 on the same split:** mirror the workflow with `keras.applications.resnet50.ResNet50` and **`resnet50.preprocess_input`**; fine-tune by training layers whose names start with `conv5`. Defaults **`EPOCHS_HEAD = 50`** and **`EPOCHS_FT = 30`** match the VGG section for a fair comparison; the cell prints wall-clock time for head + fine-tune and **test accuracy** on the best `val_loss` checkpoint.
6. **Challenge — “Fine-Tuning VGG16 for Better Performance”:** structured **objectives** (feature extraction vs fine-tuning, adapting a pretrained model, measuring gains), **tasks** (baseline head, unfreeze top layers with low LR, compare val/test and overfitting), **reflection questions** (why freeze layers, smaller LR in fine-tuning, early vs late layer features), **bonus ideas** (how many layers to unfreeze, augmentation, another backbone). The notebook ends with a **Solutions** subsection keyed to **saved runs** on the small dataset (e.g. precomputed vs augmented baselines in a results table, fine-tune test accuracy **~0.980**, discussion of validation gains and overfitting, pointers to the ResNet cells).

### Practical notes

- Training VGG/ResNet with augmentation and fine-tuning is **slow on CPU** (on the order of hours for full epoch counts); use a GPU or lower **`EPOCHS_HEAD`** / **`EPOCHS_FT`** for smoke tests.
- Saved models go under **`./models/`** (e.g. checkpoints for feature extraction, **`fine_tuning.keras`**, **`resnet50_fine_tuning.keras`**). Create `models` if needed.
- The ResNet bonus cell includes **its own imports** so it can be run after a kernel restart if the dataset path still resolves.

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
