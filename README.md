# EuroSAT Land Use Classification — Deep Learning Project

**Team 23** · Data & AI 5 — Machine Learning Project  
*Yiannis Ftiti · Ervin Lepic · Maria Miron Gavril*

---

## Overview

This project tackles multi-class land use classification on the [EuroSAT dataset](https://github.com/phelber/EuroSAT) — a collection of 27,000 satellite images across 10 land cover categories. We compare pretrained transfer learning models against a CNN trained from scratch, perform hyperparameter tuning, apply Grad-CAM for explainability, and deploy an interactive web interface using Gradio.

## Dataset

**EuroSAT** — 27,000 labeled satellite images (64×64 px, RGB)

| Class | Description |
|---|---|
| AnnualCrop | Seasonal cropland |

| Forest | Dense forest cover |

| HerbaceousVegetation | Low, non-woody vegetation |

| Highway | Roads and motorways |

| Industrial | Industrial facilities |

| Pasture | Grazing land |

| PermanentCrop | Vineyards, orchards, etc. |

| Residential | Urban/suburban housing |

| River | Flowing water bodies |

| SeaLake | Standing water bodies |

**Split:** 70% train / 15% validation / 15% test (stratified per class)

---

## Models

| Model | Type | Params frozen? |
|---|---|---|

| ResNet-50 | Pretrained (ImageNet) | Backbone frozen, FC replaced |

| ResNet-18 | Pretrained (ImageNet) | Backbone frozen, FC replaced |

| EfficientNet-B0 | Pretrained (ImageNet) | Backbone frozen, classifier replaced |

| SimpleCNN | From scratch | — |

---

## Results

| Model | Test Accuracy |
|---|---|

| **ResNet-50** | **92.79%** |

| EfficientNet-B0 | 91.43% |

| ResNet-18 | 90.20% |

| CNN from Scratch | 86.10% |

Hardest class: **HerbaceousVegetation** (~76% recall), frequently confused with Pasture and PermanentCrop.

### Hyperparameter Tuning (Learning Rate — ResNet-50)

| Learning Rate | Final Val Accuracy |
|---|---|

| 0.0001 | ~90.56% |

| **0.001** | **~92.79%** |

| 0.01 | ~90.76% |


LR = 0.001 gave the best balance of convergence speed and stability.

---

## Project Structure

```
├── Image_Learning.ipynb       # Main notebook (Colab)
├── saved_models/
│   ├── resnet18_baseline/     # Checkpoints + TensorBoard logs
│   ├── resnet50_baseline/
│   ├── efficientnet_b0_baseline/
│   ├── cnn_scratch/
│   └── resnet50_lr_*/         # Hyperparameter tuning runs(NOT PRESENT IN THIS REPOSITORY DUE TO THEIR SIZE)
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision tqdm matplotlib numpy pandas scikit-learn seaborn gradio opencv-python tensorboard
```

### Running in Google Colab

1. Mount your Google Drive and update `BASE_PATH` in the notebook to point to your shared drive folder.
2. Download the EuroSAT dataset and place it under `BASE_PATH/EuroSAT`.
3. Run all cells top to bottom. Trained model weights are saved to `BASE_PATH/saved_models/`.

### Web Interface (Gradio)

At the end of the notebook, a Gradio app is launched locally:

```python
demo.launch(share=False, inbrowser=True)
```

Upload any satellite image and get:
- Top-5 class confidence scores
- Grad-CAM heatmap overlay showing which image regions drove the prediction

---

## Explainability: Grad-CAM

We apply Gradient-weighted Class Activation Mapping (Grad-CAM) to all three pretrained models. Hooks are attached to the final convolutional layer of each architecture:

- **ResNet-50**: `layer4[-1].conv3`
- **ResNet-18**: `layer4[-1].conv2`
- **EfficientNet-B0**: `features[-1]`

Correct predictions generally show focused, semantically meaningful attention regions. Misclassified images tend to show scattered or displaced attention.

---

## Extra: GAN-based Data Augmentation

To address poor performance on HerbaceousVegetation (the worst-performing class), we trained a **DCGAN** on that class's images and generated 1,000 synthetic samples. ResNet-18 was then retrained on the augmented dataset.

**Result:** The augmented model did *not* improve accuracy on that class (accuracy dropped from 76.2% → 65.3%), suggesting the GAN-generated images introduced noise rather than useful signal. This is an interesting negative result worth investigating further.

---

## Key Takeaways

- Transfer learning provides a large advantage (+4–7%) over training from scratch, even on satellite imagery far from ImageNet's domain.
- ResNet-50 is the best overall model; EfficientNet-B0 is the best efficiency–accuracy trade-off.
- LR = 0.001 is optimal for fine-tuning a frozen backbone's classification head.
- GAN augmentation for minority/difficult classes is promising in theory but requires careful quality control in practice.

---

## Future Work

- **Progressive unfreezing** — gradually unfreeze deeper layers during training
- **Class-weighted loss** — penalize confusion between visually similar vegetation classes
- **Satellite-specific augmentations** — random rotation, multiscale cropping, atmospheric simulation
- **Better GAN training** — more epochs, FID-based quality filtering of synthetic images