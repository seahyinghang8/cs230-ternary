# CS230 Ternary Operators
Our project is to test the robustness of Ternary Weight Networks (TWN) VS. Full Precision (FP) floating point models against image noise. We will be testing our hypothesis using ResNet20 on CIFAR-10 dataset. 

We would like to credit our models to Zhu et al. for the ternary models.
```
@article{zhu2016trained,
  title={Trained Ternary Quantization},
  author={Zhu, Chenzhuo and Han, Song and Mao, Huizi and Dally, William J},
  journal={arXiv preprint arXiv:1612.01064},
  year={2016}
}
```

As part of the code is obtained from a 3rd party source with the condition of not disclosing the code to the public, thus the code is hidden away from public. Please contact us if you are interested in finding out more about the project

Change to the cifar-10 folder
```
cd cifar-10
```

To generate random noise, run
```
python random-noise-generator.py
```

To generate salt and pepper noise, run
```
python salt-and-pepper-noise-evaluator.py
```

To generate adversarial noise, run
```
python adversarial-noise-generator.py
```

To evaluate the model against a clean test set, run
```
python eval.py
```

# Misc Code
We have created a imagenet crawler and resizer in the archive/crawler folder
