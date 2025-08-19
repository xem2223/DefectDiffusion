본 프로젝트는 제조업에서 데이터 부족 문제를 해결하기 위해 설계된 조건 기반 결함 이미지 생성(Conditional Defect Synthesis) 시스템입니다.
실제 불량 데이터를 확보하기 어려운 제조 현장의 특성을 고려하여, Stable Diffusion + ControlNet + LoRA Fine-tuning을 결합하여 고품질의 NG(불량) 이미지를 합성합니다.

✅ 정상(OK) 이미지 + 결함 Mask → 현실적인 NG 이미지 생성

✅ 제조 도메인 결함(스크래치, 눌림, 벗겨짐 등) 반영 가능

✅ 웹 기반 접근성 제공 (현장 기술자도 사용 가능)

<적용 기술>

Stable Diffusion: Latent diffusion 기반 이미지 생성 모델

ControlNet: Mask/조건 입력을 활용한 세밀한 이미지 제어

LoRA Fine-tuning: 제조 데이터셋 기반 경량 파인튜닝 기법

AWS EC2: 클라우드 GPU 환경에서 학습 및 추론 수행

Django + React: 웹 기반 인터페이스 제공

<프로젝트 구조>
DefectDiffusion/
│── Dataset.py         # DefectSynthesisDataset 정의
│── train.py           # 학습 스크립트 (LoRA/ControlNet Fine-tuning)
│── infer.py           # Inference 스크립트 (조건 기반 이미지 합성)
│── 
│── models/            # Adapter, UNet, LoRA 모듈
│── configs/           # 학습/추론 설정 파일 (YAML)
│── app.py             # Django 기반 백엔드 서버
│── README.md          # 프로젝트 설명 파일
