## Advanced Python(numpy, pandas, torch) and LLM study

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

> 키워드 :  
> KNN, SVN, 결정트리, 선형희귀, 로지스틱 회귀
>
> ex) 비지도 학습이 아닌것은?

- DBSCAN / PCA
    - 밀도 기반으로 군집을 분석하고 시각화 후 하이퍼 파라미터를 변경하여 나타나는 현상 확인
        - 하이퍼 파라미터를 큰 폭으로 변경하니 클러스터의 많은 부분이 무시됨
    - 차원이 축소된 데이터 핸들링
    - 범례 및 기타 matplotlib 구성

</details>


---
<details>
    <summary><b>Day 4 - 2023-09-07</b></summary>

### 간단 머신러닝 개념
- 지도
    - KNN
        - 입력된 값이 훈련된 값의 집합과 인접한지 비교함
    - SVM
        - 데이터의 집합 사이에 선을 그어 구분하는데 선의 margin을
          gamma과 c(cost)를 조절하여 결정함.
    - 결정트리
    - 회귀
        - Iris 꽃의 종류, 타이타닉 생존자 등
    - 선형 회귀
        - 말그대로 선만 긋기떄문에 속도가 빠르지만 정확도가 떨어짐
    - 로지스틱 회귀
        - 곡선을 그릴 수 있으며, 당연히 속도가 느려지고 정확도가 비교적 높음
- 비지도
    - 계층 군집화(Hierarchical Clustering)
        - 개별 개체들을 하나의 클러스터로 보고, 가까운 클러스터끼리 합치면서 클러스터의 개수를 줄여 나가는 방식입니다.
    - DBSCAN
        - 밀도 기반의 군집화 알고리즘으로, 밀도가 높은 부분을 클러스터로 인식합니다.
    - PCA (Principal Component Analysis)
        - 다차원의 데이터를 시각화하거나 차원을 축소할 때 주로 사용되는 비지도학습 방법입니다.
### CNN, DNN 코드 코멘트
- 'FashionMNIST'을 처리하는 CNN과 DNN 모델 작성, 각 epoch당 진행 상황 (iteraction, loss, accuracy)을 출력시켜 학습 과정을 확인.
- CNN은 데이터가 약간만 달라져도 정확도가 떨어지기에 의미가 없음.
- DNN은 데이터가 달라져도 정확도가 높은 편임.
하지만 학습 데이터에 한해선 CNN과 DNN 모두 iteration이 2만까지 늘어나도 정확도는 비슷했음 (CNN 89%, DNN 90%)

### 전이학습 - Transfer Learning

- 미리 학습된 모델을 가져와서 사용함.
- 데이터셋을 불러와 `전처리 -> 모델 불러오기 -> 최적화/손실 함수 선언 -> 추가 학습 -> 테스트` 과정을 거쳐 모델을 추가학습 시키는 코드를 작성함.
- 테스트를 거쳐 예측 결과를 기반으로 손실을 계삲고 최적화 과정을 거쳐 epoch를 반복함.
    - 최고의 정확도를 가지는 모델을 저장함.
</details>

---