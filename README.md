### Pytorch Lightning template
Use this template to rapidly bootstrap a DL project with:

- Built in `setup.py`
- Using `conda` to manage Python in `environment.yaml` 
- Examples with MNIST
- Badges
- Bibtex

First, install dependencies   
```bash
# clone project   
git clone https://github.com/username/pytorch-lightning-template

# install project   
cd pytorch-lightning-template
conda env update -f environment.yaml
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so into your entry point scripts:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

Execute your entry scripts like:
```bash
python train.py    
# or
python test.py
```

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>
 
## Description   
ConSelfSTransDrLIB: Contrastive Self-supervised Transformer Disentangled Representation Learning with Inductive Biases is All you need, and where to find them.

## How to run 
```bash
python abracadabra.py
```


### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
