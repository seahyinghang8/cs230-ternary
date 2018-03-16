# CS230 Ternary Operators
We are utilizing cs230-code-examples provided in tensorflow/vision to study the impact on performance by ternarizing the weights.

Install the hand sign dataset from the starter code, and follow the same procedure as stated on the started code.

To ternarize the base_model, run
```
python ternarize.py
```

To evaluate the model, run
```
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/ternary
```

# Misc Code
We have created a imagenet crawler and resizer in the /crawler folder
