# Fine tunning of a ViT transformers model for image classification

ViT is a transformers model that can be used for image classification.
This repository will show you how to fine tune this model with your own database.

## Requirements

### Python dependencies

First you need to install all python libraries :

```bash
python -m pip install -r requirements.txt
```

### Data folder structure

Create 3 subfolders : `test`, `train`, `val`.

Into this three folders, create a folder for each category of your data.

## Usage

```bash
python runner.py
```
