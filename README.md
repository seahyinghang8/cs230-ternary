# CS230 Ternary Operators
Our project is to test the robustness of ternary network models VS. full precision models against image noise. We will be testing our hypothesis using ResNet20 on CIFAR-10 dataset. 

We would like to credit our models to Zhu et al. for the ternary models.
```
@article{zhu2016trained,
  title={Trained Ternary Quantization},
  author={Zhu, Chenzhuo and Han, Song and Mao, Huizi and Dally, William J},
  journal={arXiv preprint arXiv:1612.01064},
  year={2016}
}
```

We will test our network against random noise and adversarial noise.

To generate random noise, run
```
cd cifar
python random-noise-generator.py
```

To generate adversarial noise, run
```
python adversarial-noise-generator.py
```

To evaluate the model, run
```
python eval.py
```

# Misc Code
We have created a imagenet crawler and resizer in the /crawler folder
