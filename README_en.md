# The AI Study Diary

## Overview

"The AI Study Diary" chronicles a journey through Artificial Intelligence and Machine Learning,
using Python and PyTorch.
The project encompasses a variety of topics including tensor operations on GPUs,
machine learning techniques
like Support Vector Machines (SVM), logistic regression, PCA, DBSCAN,
and applying these concepts in PyTorch for tasks such as image classification.
The diary is a practical resource for anyone keen to delve into Machine Learning and AI.

---

<details>
    <summary><b>Day 1 - 2023-09-04</b></summary>


> Already know all of these :(

- Git / GitHub usage
    - README.md
        - Markdown basic syntax
    - Edit file on GitHub
- Object oriented programming
    - Special method
    - Extend class and `super()`

#### Scala? Vector? Tensor?

- Scala [x]
- Vector [x, y]
- Tensor [x, y, ...z]

#### On GPU

```python
import torch

# !!! Before !!!
print(torch.cuda.is_available())  # It must be True

ex = torch.tensor([[1, 2], [3, 4]], device="cuda:0")  # cuda:n is index of GPU
res = ex.to("cpu").numpy()
print(res)
```

#### Controlling Shape

```python
import torch

a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)
b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

c = a + b

print(c.shape)
print(c.view(8, 1))
print(c.view(1, 8))
```

</details>

#### Advanced Python and basic of tensor

---

<details>
    <summary><b>Day 2 - 2023-09-05</b></summary>


---

### 코드 목차

- Numpy
    - Array
    - Indexing
    - To Tensor
- Pandas
- Matplotlib
- Car Evaluation Dataset (w. PyTorch)
    - Data
        - Preprocessing
        - Visualization
    - Model
        - Training
        - Evaluation

</details>

#### Handing car evaluation dataset train/eval and visualize

---
<details>
    <summary><b>Day 3 - 2023-09-06</b></summary>

- Pandas
    - DataFrame
- Re-learn basic machine learning concepts
- Support Vector Machine `(SVM)`
- `Nonlinear` and `linear classification`
- Predict number using `logistic regression`
- About `Confusion Matrix`

*Linear classification* is faster than *non-linear classification*, but if data is not linearly distributed, linear
regression cannot be used.
In this case, you need to use *non-linear regression*.

> Keywords:
> KNN, SVN, Decision Tree, Linear Regression, Logistic Regression
>
> ex) Which of the following is not unsupervised learning?

- DBSCAN / PCA
    - Analyze and visualize clusters based on density, then observe the phenomenon that occurs when changing the
      hyperparameter
        - A large part of the cluster is ignored when the hyperparameters are significantly changed
    - Handling data with reduced dimensions
    - Legends and other matplotlib configurations

</details>

#### SVM, Logistic Regression, Confusion Matrix

---
<details>
    <summary><b>Day 4 - 2023-09-07</b></summary>

### Simple Machine Learning Concepts

- Supervised
    - KNN
        - It compares whether the input value is adjacent to the trained values' set.
    - SVM
        - It draws a line between the sets of data to distinguish them. Gamma and
          c(cost) adjust the margin of the line.
    - Decision Tree
    - Regression
        - Types of Iris flower, Titanic survivors, etc.
    - Linear Regression
        - As it literally only draws a line, it is faster but less accurate.
    - Logistic Regression
        - It can draw curves, so it's naturally slower but relatively more accurate.
- Unsupervised
    - Hierarchical Clustering
        - It views individual objects as one cluster and merges the nearby clusters, reducing the number of clusters.
    - DBSCAN
        - A density-based clustering algorithm, which recognizes the high-density part as a cluster.
    - PCA (Principal Component Analysis)
        - It is a commonly used unsupervised learning method for visualizing or reducing the dimension of
          multidimensional data.

### Comments on CNN, DNN code

- Writing CNN and DNN models that process 'FashionMNIST,' and print the progress (iteration, loss, accuracy) of each
  epoch to monitor the learning process.
- It is meaningless for CNN because the accuracy drops even with slight data variations.
- DNNs maintain relatively high accuracy even if the data changes.
  However, for learning data, both CNN and DNN had similar accuracy even when the iteration increased up to 20,000 (CNN
  89%, DNN 90%)

### Transfer Learning

- Using pre-trained models.
- The code is written to further train the model by going through the process
  of `load dataset -> preprocessing -> load model -> declare optimization/loss function -> additional learning -> test`.
- After testing, it calculates the loss based on prediction results, goes through the optimization process and repeats
  epochs.
    - It saves the model with the highest accuracy.

</details>

#### Hierarchical Clustering, DBSCAN, PCA

---


<details>
    <summary><b>Day 5 - 2023-09-11</b></summary>

# Image classifier using pretrained ResNet models

### Why is jupyter needed?

- Jupyter is based on IPython (Interactive Python)
- Basically, a once executed Python script is gone at the end of the execution.
- Jupyter allows you to keep the output of a Python script and re-run it later. (Reside in memory)
- Machine learning code usually takes a lot of time by one function call.
    - So, we save the output of the functon and save the time.

### Cat and dog classification using pretrained ResNet models

- Loads cat and dog images from training data
- Utilize ResNet model that has been pretrained for image classification
- Applies transformations on the dataset for further efficiency
- Customizes the last layer of the model to suit the two classes (cat and dog)
- Defines a custom training function `train_model` which iterates over the dataset for a given number of epochs
- Within `train_model`, it adjusts model weights based on calculated loss and tracks the best model state
- Save state of the best model that can be loaded for later use

### Image evaluation using saved models

- After the training process, the `eval_model` function is used to evaluate the model performance on the test dataset
- All saved models during training are loaded, and the model's prediction accuracy is evaluated
- The model with the best accuracy is identified

</details>

#### Image classifier using pretrained ResNet models

---

<details>
    <summary><b>Day 6 - 2023-09-12</b></summary>

- The `ImageTransform` utility class, which normalizes all images, is used to uniformly change the size of the photos
  and separate the training(train) and validation(valid) data.
    - To prevent overfitting the direction of the data, you double the training data by flipping half of the images.
        - During the validation process, `RandomHorizeontalFlip()` is not used because rotation is not necessary.
- Since there is too much training data, only 400 photos are used for training.
- In the loading process, the `os.path.join()` function is used to correctly fetch the paths by merging them.
- The code `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` in the code that loads the dataset is a process to convert the color
  because OpenCV uses BGR values, not RGB. (Following the example in the book that loads images with `cv2`, this is not
  an efficient method.)
- Set the label name as the name of the subdirectory of the training data folder.
    - In this process, because the separator is different depending on the operating system, use `os.path.sep`.
        - `abel = img_path.split(path.sep)[-len(path.sep)].split('.')[0]`
- As a result of training, the accuracy was not high, but it showed meaningful results.

</details>

#### Is cat or dog? (Image classifier using LeNet)

## License

> This repository contains code samples from the
> book [GilbutITBook](https://github.com/gilbutITbook/080289).  
> Some of my code, including the README (which identifies me), is subject to the MIT License.
