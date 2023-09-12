# The AI Study Diary

## 개요

이 리포지토리는 Python과 PyTorch를 사용하여 머신러닝의 기초를 학습하며 사용한 코드와 개념을 기록합니다.
GPU의 텐서 작업, SVM(Support Vector Machines)과 같은 기계 학습, 로지스틱 회귀, PCA, DBSCAN, 이미지 분류와 같은 작업을 위해
PyTorch를 사용해 코드를 작성합니다. 수업 기록이므로, 목차에 대해 모든 코드가 포함되어 있지 않을 수 있습니다.


<details>
    <summary><b>Day 1 - 2023-09-04</b></summary>

- OOP 기본 개념
    - 스페셜 메소드
    - `super()` 및 클래스 상속

#### 스칼라? 벡터? 텐서?

- Scala [x]
- Vector [x, y]
- Tensor [x, y, ...z]

#### GPU에서의 PyTorch

```python
import torch

# !!! Before !!!
print(torch.cuda.is_available())  # 참이여야 합니다.

# cuda:n 형식으로 작성합니다. n은 GPU의 인덱스 번호입니다.
ex = torch.tensor([[1, 2], [3, 4]], device="cuda:0")
res = ex.to("cpu").numpy()
print(res)
```

#### Shape 조작하기

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

#### 고급 파이썬 및 텐서

---

<details>
    <summary><b>Day 2 - 2023-09-05</b></summary>


---
### 코드 목차
- Numpy
    - 배열
    - 인덱싱
    - 배열을 텐서로 변환하기
- Pandas
- Matplotlib
- 자동차 평가 데이터 세트 (w. PyTorch)
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
    - 데이터프레임
- 기본적인 머신러닝 개념 복습하기
- 서포트 벡터 머신 `(SVM)`
- `비선형`과 `선형 분류`
- `로지스틱 회귀`를 사용한 숫자 예측
- `혼동 행렬`에 대하여

*선형 분류*는 *비선형 분류*보다 빠르지만, 만약 데이터가 선형적으로 분포되어 있지 않다면, 선형 회귀는 사용될 수 없습니다.
이 경우, *비선형 회귀*를 사용해야만 합니다.

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

#### SVM, Logistic Regression, Confusion Matrix

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

#### Hierarchical Clustering, DBSCAN, PCA

---


<details>
    <summary><b>Day 5 - 2023-09-11</b></summary>

# 사전훈련된 ResNet 모델을 사용하여 이미지 분류하기

### 왜 주피터가 필요한가요?

- 주피터는 IPython (Interactive Python) 기반입니다.
- 기본적으로, 한번 실행된 파이썬 스크립트는 실행이 끝나면 사라집니다.
- 주피터를 사용하면 파이썬 스크립트의 출력을 유지하고 나중에 다시 실행할 수 있습니다. (메모리에 유지)
- 머신 러닝 코드는 보통 한 함수 호출로 인해 많은 시간을 소모합니다.
    - 따라서, 함수의 출력을 저장하여 시간을 아낄 수 있게 됩니다.

### 사전 훈련된 ResNet 모델을 사용한 고양이와 개 분류

- 훈련 데이터로부터 고양이와 개 이미지를 로드합니다.
- 이미지 분류를 위해 사전 훈련된 ResNet 모델을 활용합니다.
- 효율성을 높이기 위해 데이터셋에 변형을 적용합니다.
- 모델의 마지막 층을 두 클래스(고양이와 개)에 맞게 커스터마이징합니다.
- 주어진 에포크 수 동안 데이터셋을 반복하는 사용자 정의 훈련 함수 `train_model`을 정의합니다.
- `train_model` 내에서, 계산된 손실에 기초하여 모델 가중치를 조정하고 최상의 모델 상태를 추적합니다.
- 나중에 사용할 수 있도록 최상의 모델 상태를 저장합니다.

### 저장된 모델을 사용한 이미지 평가

- 훈련 과정 후에는 `eval_model()` 함수를 사용하여 테스트 데이터 세트에 대한 모델 성능을 평가합니다.
- 훈련 중에 저장된 모든 모델을 로드하고 모델의 예측 정확도를 평가합니다.
- 정확도가 가장 높은 모델을 식별하여 저장합니다.

</details>

#### 사전 학습된 ResNet 모델을 사용하는 이미지 분류기

---

<details>
    <summary><b>Day 6 - 2023-09-12</b></summary>

- 모든 이미지를 정규화하는 `ImageTransform` 유틸리티 클래스를 사용하여 사진의 크기를 일괄되게 변경하고, 학습(train)과 검증(vaild) 데이터를 분리합니다.
    - 데이터의 방향에 과적합 되지 않도록 이미지의 절반을 뒤집어서 학습 데이터를 늘립니다.
        - 검증 과정에선 회전이 필요 없으므로 `RandomHorizeontalFlip()`을 사용하지 않습니다.
- 학습 데이터가 너무 많으므로 400개의 사진만 학습용으로 사용합니다.
- 불러오는 과정에서 `os.path.join()` 함수를 사용하여 경로를 합쳐 정확하게 불러옵니다.
- 데이터셋을 불러오는 코드 중 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 라는 코드는 OpenCV가 RGB가 아닌 BGR 값을 사용하기 때문에 색상을 변환하기 위한 과정입니다. (책의
  예제에서 `cv2`로 이미지를 불러오기에 따라했지만 효율적이지 못한 방법입니다.)
- 라벨의 이름을 학습 데이터 폴더의 하위 디렉토리 이름으로 설정합니다.
    - 이 과정에서 운영체제에 따라 separator가 다르므로 `os.path.sep`을 사용합니다.
        - `abel = img_path.split(path.sep)[-len(path.sep)].split('.')[0]`
- 학습 결과, 정확도가 높진 않지만 유의미한 결과를 보여주었습니다.

</details>

#### 강아지와 고양이 구별하기 (LeNet)

## License

> 이 저장소는 [GilbutITBook](https://github.com/gilbutITbook/080289) 책의
> 코드 샘플들을 포함하고 있습니다.
> 일부 코드, README 포함 (개인을 식별하는 정보를 담고 있음)은 MIT 라이선스에 따릅니다.