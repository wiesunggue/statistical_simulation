# 4.6 예측 정확도의 평가와 변수 선택

# 정합도와 예측 정확도
# 정합도는 가지고 있는 데이터에 대해 모델을 적용했을 때 들어맞는 정도
# 예측 정확도는 아직 얻지 못한 데이터에 대해 모델을 적용했을 때 들어맞는 정도

# 과적합: 적합도는 높은데 예측 정확도가 낮아지는 경우 -> 지나치게 적합한 모델을 구성함
# 필요 없는 변수를 제외하는 것만으로도 예측 정확도가 더욱 높아지게 될 가능성이 존재

# 아직 얻지 못한 데이터에 대한 오차 : 일반화 오차 -> 고려하지 못한 변수에 의한 예측 오차가 존재함

# 훈련 데이터와 테스트 데이터
# 훈련 데이터는 파라메터 추정에 사용되는 데이터 -> 적합도만 구할 수 있고, 예측 정확도는 알 수 없음
# 테스트 데이터는 훈련에 사용하지 않고 남겨둔 데이터 -> 파라메터 추정에 사용되지 않아 예측 정확도를 구할 수 있음

# 교차 검증 -> 데이터를 훈련 데이터와 테스트 데이터로 나누어 테스트 데이터에 대한 검증 진행
# 교차 검증 기법으로 리브-p-아웃 교차검증, K겹 교차검증 이 있음
# 리브-p-아웃 교차검증 : p개의 데이터를 추출하고 남은 데이터를 테스트 데이터로 사용하는 방법
# K겹 교차검증 : 데이터를 K개의 그룹으로 나누어 그 그룹에서 하나를 추출하여 테스트 데이터로 사용하고 이것을 K번 반복하여 예측 정확도의 평균값을 구함

# AIC 아카이케 정보 기준
# AIC = -2*(최대 로그우도-추정 파라메터의 수)
# AIC가 작을 수록 좋은 모델임
# AIC는 모델의 적합도를 높이게만 하는 것이 아니라 파라메터의 수가 적으면서 모델의 적합도가 좋게 하는 것이라 조금 더 나음
# AIC는 계산량이 적음

# 상대 엔트로피
# 평균로그우도?

# 검정 대신 변수 선택
# 모델 1: 체온~독립변수 없음
# 모델 2: 체온~약의 유무
# 모델1,2의 AIC결과 모델 2가 더 작다면 모델 2를 쓰는 것이 좋음