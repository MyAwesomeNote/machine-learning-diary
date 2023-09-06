## Advanced Python(numpy, pandas, torch) for LLM study

### Day1 - 2023-09-04

---
> Already know per: 100%

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

### Day2 - 2023-09-5

---
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

### Day3 - 2023-09-06

---

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

> 키워드 :  
> KNN, SVN, 결정트리, 선형희귀, 로지스틱 회귀
>
> ex) 비지도 학습이 아닌것은?

- DBSCAN / PCA
    - 밀도 기반으로 군집을 분석하고 시각화 후 하이퍼 파라미터를 변경하여 나타나는 현상 확인
        - 하이퍼 파라미터를 큰 폭으로 변경하니 클러스터의 많은 부분이 무시됨
    - 차원이 축소된 데이터 핸들링
    - 범례 및 기타 matplotlib 구성