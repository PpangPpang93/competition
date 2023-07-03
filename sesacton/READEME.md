### 새싹톤 참가
- 2023.06.16 발표

### 대회 요약
- Notion 참고 : https://www.notion.so/9669370a93c8442791dce5ebffe93f07?pvs=4

### 구현 기능
- 연산량이 과하게 많은 이미지 캡셔닝 모델을 사용하지 않고, 캡셔닝과 유사한 기능을 구현
- teachable machine으로 이미지 분류 모델 + opencv 클러스터링을 활용한 색상 분류 + 도메인 정보를 활용한 세부 정보 추정

#### 구현 정보
- 파일 : sesacton/model/inference.py
    - classification
        - inference에 사용한 것 : teachable machine 서비스
        - 시도한 것 : ResNet custom train(sesacton/model/train.py)
    - color : color_detection.py
    - 세부 정보 : 생략
