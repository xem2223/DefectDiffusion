본 프로젝트는 제조업에서 데이터 부족 문제를 해결하기 위해 설계된 조건 기반 결함 이미지 생성(Conditional Defect Synthesis) 시스템입니다.
실제 불량 데이터를 확보하기 어려운 제조 현장의 특성을 고려하여, Stable Diffusion + ControlNet + LoRA Fine-tuning을 결합하여 고품질의 NG(불량) 이미지를 합성합니다.

✅ 정상(OK) 이미지 + 결함 Mask → 현실적인 NG 이미지 생성

✅ 제조 도메인 결함(스크래치, 눌림, 벗겨짐 등) 반영 가능

✅ 웹 기반 접근성 제공 (현장 기술자도 사용 가능)


<적용 기술>

Stable Diffusion: Latent diffusion 기반 이미지 생성 모델

ControlNet: Mask/조건 입력을 활용한 세밀한 이미지 제어

LoRA Fine-tuning: 제조 데이터셋 기반 경량 파인튜닝 기법

Latent Precomputation: VAE를 통해 OK 이미지/Mask의 latent & cond를 사전에 계산하여 학습 효율 극대화

AWS EC2: 클라우드 GPU 환경에서 학습 및 추론 수행

Django + React: 웹 기반 인터페이스 제공


<프로젝트 구조>

DefectDiffusion/

│── Dataset.py         # DefectSynthesisDataset 정의

│── Train.py           # 학습 스크립트 (LoRA/ControlNet Fine-tuning)

│── inference.py           # Inference 스크립트 (조건 기반 이미지 합성)

│── precompute_latents.py  # OK 이미지/Mask의 Latent & Condition 사전 계산

│── models/            # Adapter, UNet, LoRA 모듈

│── genweb/             # Django 기반 백엔드 서버

│── README.md          # 프로젝트 설명 파일

<필수 패키지 및 버전 정보>
<Stable Diffusion + ControlNet + LoRA + PEFT + Inpaint 파이프라인에서의 조합>

| 패키지                 | 버전     | 역할                                     |
| :------------------ | :----- | :------------------------------------- |
| **torch**           | 2.1.2  | PyTorch 핵심 딥러닝 프레임워크                   |
| **torchvision**     | 0.16.2 | 이미지 데이터셋 및 변환(Transforms) 지원           |
| **torchaudio**      | 2.1.2  | 오디오 데이터 처리용 PyTorch 확장                 |
| **diffusers**       | 0.27.2 | Stable Diffusion 및 관련 모델 프레임워크         |
| **transformers**    | 4.39.3 | 텍스트 인코딩용 모델(CLIP, T5 등)                |
| **accelerate**      | 0.27.2 | 멀티 GPU 및 분산 학습 지원                      |
| **peft**            | 0.10.0 | LoRA 등 파라미터 효율적 미세조정(PEFT) 지원          |
| **huggingface_hub** | 0.20.3 | Hugging Face Hub 연동 및 모델 관리            |
| **safetensors**     | 0.4.2  | 빠르고 안전한 모델 가중치 저장 포맷                   |
| **xformers**        | 0.0.23 | 메모리 효율적 Transformer 연산 (Attention 최적화) |


<inference.py 실행 방법>

python inference.py \
--unet_lora_dir ./models/unet_lora \
--ctrl_lora_dir ./models/ctrl_lora \
--prompt "switch contamination defect" \
--negative "unnecessary changes, altered background, unrealistic texture" \
--cond ./samples_img2img/09_switch_contamination_seed132_COND.png \
--init ./samples_img2img/09_switch_contamination_seed132_OK.png \
--out ./samples_infer --steps 32 --scale 6.0 --strength 0.3 \
--n 2 --seed 123 --fp16 --bypass_check

cond와 init은 입력 이미지 경로
prompt는 입력 텍스트 (클래스명 + 결함명)
