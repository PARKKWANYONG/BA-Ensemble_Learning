# BA-Ensemble_Learning

# Bagging Vs Boosting

해당 챕터에서는 배깅과 부스팅에 대해 설명하겠습니다.
대부분 배깅과 부스팅이라는 말은 들어보았지만 이 용어가 무엇을 의미하는지에 대해 무지합니다.
따라서 이번 챕터에서는 배깅과 부스팅의 용어에 대해 공부하고 구현해보도록하겠습니다.

1. Introduction Ensemble Learning

- 배깅과 부스팅은 Machine Learning기반의 앙상블 학습 방법입니다.

- 배깅과 부스팅은 둘 다 앙상블 기술이라는 점에서 유사하며, 약한 학습자가 결합되어 더 나은 성능을 갖는 강한 학습자를 생성합니다.

- 앙상블 학습은 여러 모델을 결합하여 기계 학습 모델 성능을 향상시키는 데 도움이 됩니다. 이 접근 방식을 사용하면 단일 모델에 비해 더 나은 예측 성능을 생성할 수 있습니다.

- 앙상블 학습의 기본 아이디어는 분류자가 집합을 학습하고 개별적으로 Votting을 할 수 있도록 하는 것입니다. 

- 머신 러닝의 다양화는 앙상블 학습이라는 기술에 의해 달성됨을 알 수 있습니다.

- 즉 앙상블의 핵심은 결과 집합을 예측하거나 분류하는 것을 목표로 하는 여러 모델을 훈련시키는 것입니다.

- 배깅과 부스팅은 앙상블 학습 기법의 두 가지 유형입니다. 이 두 가지는 서로 다른 모델의 여러 추정치를 결합하므로 단일 추정치의 분산을 줄입니다. 따라서 결과는 더 높은 안정성을 가진 모델이 될 수 있습니다.

- 학습 오류의 주요 원인은 잡음, 편향 및 분산 때문입니다. Ensemble은 이러한 요소를 최소화하는 데 도움이 됩니다. 앙상블 방법을 사용하여 최종 모델의 안정성을 높이고 오류를 줄일 수 있습니다.

- 배깅은 모델의 분산을 줄이는 데 도움이 됩니다. 부스팅은 모델의 편향을 줄이는 데 도움이 됩니다.

- 이러한 방법은 기계 학습 알고리즘의 안정성과 정확성을 향상시키기 위해 설계되었습니다. 여러 분류기의 조합은 특히 불안정한 분류기의 경우 분산을 줄이고 단일 분류기보다 더 신뢰할 수 있는 분류를 생성할 수 있습니다.


![image](https://user-images.githubusercontent.com/71392868/204717187-65da366c-3f74-45fa-bd42-65afd19c3038.png)


2. Boostrapping

- 부트스트랩은 간단히 말해 무작위 표본 추출을 의미합니다.

- Bootstrap을 사용하면 데이터 세트의 편향과 분산을 더 잘 관측 할수 있습니다.

- 따라서 부트스트래핑은 데이터 세트에서 관측값의 하위 집합을 대체하여 생성하는 샘플링 기술입니다. 

- 하위 집합의 크기는 원래 집합의 크기와 동일합니다. 

- 부트스트랩에는 데이터 세트에서 데이터의 작은 하위 집합을 무작위로 샘플링하는 작업이 포함됩니다. 이 Subset은 replacement가 가능합니다..

- 데이터세트를 추출할 확률은 모두 동일합니다. 

- 이 방법은 데이터 세트의 평균과 표준 및 편차를 파악하는 데에 도움이 됩니다. 

- 부스트랩의 직관적인 그림은 아래와 같습니다. 


![image](https://user-images.githubusercontent.com/71392868/204717735-5f1f27cc-4df5-4f57-a44e-3c61460213dd.png)


3. Bagging

- Bagging(또는 Bootstrap Aggregation)은 간단하고 매우 강력한 앙상블 방법입니다. 

- 배깅은 일반적으로 결정 트리와 같은 고분산 기계 학습 알고리즘에 부트스트랩 절차를 적용한 것입니다. 

- 배깅의 기본 개념은 일반화된 결과를 얻기 위해 여러 모델(예: 모든 의사 결정 트리)의 결과를 결합하는 것입니다. 

- 배깅은 이러한 하위 집합을 사용하여 분포(완전한 집합)에 대한  fair한 아이디어를 얻습니다. 

- 배깅을 위해 생성된 하위 집합의 크기는 원래 집합보다 작을 수 있습니다. 

- 배깅에 대한 직관적인 그림은 아래와 같습니다. 


![image](https://user-images.githubusercontent.com/71392868/204741645-a6c2613b-bcdc-4dfa-8067-d2bf718b0869.png)


3-1. Bagging의 작동방식

1. 원본 데이터 세트에서 하위 집합을 선택합니다.  

2. Base Model(Weak Model)들은 이 하위 집합에서 생성됩니다.

3. 모델은 서로서로 병렬적이게 그리고 독립적이게 작동합니다.

4. 모든 모델의 예측 값을 결합하여 최종 예측값을 반환합니다.

- 배깅의 작동 방식에 대한 직관적인 그림은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/71392868/204743273-1e29e934-5c41-4a9a-bc07-58ae54589b7b.png)


4. Boosting

- 부스팅은 순차적인 프로세스로 후속 모델은 이전 모델의 Error를 Correct 합니다.

- 후속 모델은 이전 모델에 따라 달라집니다. 이 기법에서 Early Learners가 데이터에대한 간단한 모델을 fitting 시킨 뒤에 오류가 있는지 데이터를 분석하는 방식으로 순차적이게 학습합니다.

- 즉 연속 트리(랜덤 샘플)를 적합시키고 모든 단게에서 이전 트리의 오류를 해결하는 것이 부스팅의 목표입니다.

- 특정한 데이터가 잘못 분류된다면, 다음 데이터에서 정확하게 분류할 수 있도록 가중치가 증가합니다. 

- 최종적으로, 전체 세트를 결합함으로써 Weak Learner를 Strong Learner로 변환시킵니다.

4-1. 부스팅의 작동방식

1. 원본 데이터에서 하위 집합이 생성됩니다.

2. 초기에, 모든 데이터 포인트들은 동일한 Weights를 갖습니다.

3. Base Model은 위에서 생성된 하위 집합에서 만들어집니다.

4. 전체 데이터 셋에서 예측을 수행합니다.

![image](https://user-images.githubusercontent.com/71392868/204744896-eaaab227-9a26-45d2-a0bb-df2ef7b823c2.png)

5. 예측 값과 실제 값에 의해서 Error를 계산합니다.

6.만약 관측치가 잘못 예측되었다면 높은 웨이트를 부여합니다.

7. 또 다른 모델이 생성되고 데이터에 대한 예측이 이루어집니다. (해당 모델은 이전 모델의 오류를 반영합니다.)


![image](https://user-images.githubusercontent.com/71392868/204745939-d3fc42db-de08-4626-96ea-9a675ce93726.png)


8. 이러한 방식으로 여러 모델이 생성되었다면

9. 최종적인 모델 (Strong Learner)은 모든 Weak Learner의 Weight가 반영되어있습니다.

![image](https://user-images.githubusercontent.com/71392868/204746220-a7034e66-ed0d-4692-b9b5-1c5740439daa.png)

- 따라서 부스팅 알고리즘은 다수의 약한 학습자를 결합하여 강력한 학습자를 형성합니다.

- 각 모델은 앙상블 모델(Strong Learner)의 성능을 향상시킨다.


References

https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/

https://medium.com/swlh/difference-between-bagging-and-boosting-f996253acd22

https://www.geeksforgeeks.org/comparison-b-w-bagging-and-boosting-data-mining/

https://hub.packtpub.com/ensemble-methods-optimize-machine-learning-models/

https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60f


