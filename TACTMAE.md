# TactMAE (TacOctMAE)

**촉각 정보를 이용한 옥트리 기반 피드 포워드 3차원 재구성 모델**
김재훈\*, 송지훈\*, 김범준, 유재준† (UNIST 인공지능대학원)

단일 RGB-D 이미지에 희소한 촉각 접촉점을 추가 입력으로 넣어, 처음 보는 물체의 전체 3D 형상을 한 번의 forward pass로 복원하는 모델. 옥트리 기반 zero-shot 형상 복원 모델 OctMAE (Iwase et al., ECCV 2024)를 기반으로 한다.

---

## 1. 왜 필요한가

**시각만으로는 가려진 영역을 알 수 없다.**
단일 시점 RGB-D는 카메라를 향한 표면만 관측한다. 물체의 뒷면, 바닥, 손잡이처럼 가려진 영역은 기하 정보가 근본적으로 없어서, 처음 보는 물체일수록 정확히 복원하기 어렵다. 시점을 더 늘려 보완하려 해도 이미지 처리 비용 때문에 실시간 갱신이 어렵다.

**촉각은 가려진 영역의 실제 표면을 직접 준다.**
촉각 센서의 접촉점은 물체 표면에 실제로 존재하는 점의 3D 위치다. 카메라에서 보이지 않는 곳도 측정할 수 있어 시각 복원의 사각지대를 직접 메운다.

**기존 시각-촉각 방법은 느리다.**

| 방법 | 표현 | 추론 |
|---|---|---|
| TouchSDF (RA-L 2024) | SDF (DeepSDF) | 추론 때 latent를 최적화 → 물체 하나에도 오래 걸림 |
| Touch2Shape (CVPR 2025), TouchDiffusion (RA-L 2026) | Diffusion | 여러 단계 반복 샘플링 → 추론 시간이 김 |

로봇이 실시간으로 인식하고 행동해야 하는 환경에서는 이 추론 비용이 큰 제약이 된다.

**옥트리 기반이면 빠르고, 촉각을 직접 넣을 수 있다.**
- OctMAE는 옥트리 위에서 표면을 직접 예측하므로, 후처리나 반복 샘플링 없이 단일 forward pass로 복원한다.
- 옥트리는 3D 공간을 명시적인 voxel 좌표로 나누므로, 촉각 접촉점을 해당 voxel에 바로 대응시켜 입력 단계에서 결합할 수 있다.
- 반면 암시적 표현(SDF, diffusion)은 촉각을 조건부 임베딩이나 추가 손실로 간접적으로 학습해야 해서 해석과 제어가 어렵다.

**주요 기여**
1. OctMAE에 희소한 촉각 접촉점을 추가 입력으로 통합
2. 반복 샘플링, 후처리 없이 단일 forward pass로 복원 (실시간 가능)
3. 물체당 약 20개 접촉만으로 가려진 영역 복원 품질이 일관되게 향상됨을 ShapeNet에서 확인

---

## 2. 데이터 생성

동일한 물체에 대해 RGB-D 관측과 촉각 접촉점을 모두 가진 데이터가 필요하다. TouchSDF의 수집 절차를 따른다.

| 단계 | 내용 |
|---|---|
| 물체 | ShapeNet 6개 카테고리 (Bottle, Mug, Bowl, Camera, Guitar, Jar), 2,404개 |
| 촉각 (3D) | PyBullet 기반 Tactile Gym + TacTip 센서. 물체를 감싸는 가상의 구 표면에서 접촉 위치를 균일하게 무작위 샘플링하고, 센서를 물체 중심 방향으로 정렬해 같은 속도와 힘으로 누름. 물체당 접촉 20번, 접촉마다 접촉 패치를 나타내는 점 200개 |
| 시각 (RGB-D) | BlenderProc으로 물체 주위 48개 시점에서 렌더링 |
| 정렬 | 촉각 점과 RGB-D를 같은 좌표계로 변환해 시각-촉각 쌍 구성 |

실행 순서: TouchSDF `data_making/extract_touch_charts.py` (촉각) → `scripts/precompute_tactile.py` (카메라 좌표계 변환) → `scripts/make_shards.py` (webdataset shard). 데이터 통계는 [DATASET.md](DATASET.md).

---

## 3. 구조

```
RGB ──► ResNeXt-50 이미지 인코더 ──► 픽셀 feature ──┐
Depth + 물체 마스크 ──► 3D로 역투영 ─────────────────┤
                                                    ▼
촉각 접촉점 (20 × 200 = 4,000점) ──► 학습 가능한 촉각 토큰
                                                    │
                       입력 옥트리 feature (LoD-9) ◄─┘
                                │
                        Octree U-Net 인코더
                                │
                        잠재 옥트리 feature (LoD-5)
                                │
              3D MAE 인코더 → 3D MAE 디코더 (가려진 voxel에 mask token)
                                │
                     재구성된 옥트리 feature (LoD-5)
                                │
                        Octree U-Net 디코더 (LoD-5 → 9)
                                │
                                ▼
                          3D 표면 (LoD-9)
```

LoD-h = 한 변을 2^h 등분한 해상도. LoD-9는 512, LoD-5는 32.

### 3-1. OctMAE (베이스 모델)

| 구성 | 역할 |
|---|---|
| 이미지 인코더 | 사전 학습된 ResNeXt-50으로 RGB에서 고해상도 feature 추출 |
| 3D 역투영 | depth와 전경 마스크로 feature를 3D 공간에 올려 LoD-9 옥트리 feature 구성 |
| Octree U-Net 인코더 | LoD-9 → LoD-5 잠재 표현으로 압축 |
| 3D MAE | 가려져 정보가 없는 voxel 위치에 학습 가능한 mask token을 넣고, 관측된 voxel feature와 함께 처리해 가려진 영역의 전역 형상을 추론 |
| Octree U-Net 디코더 | LoD-5 → LoD-9로 점진적으로 올리며 표면을 직접 예측 |

### 3-2. 촉각 통합 (TactMAE에서 추가)

- **접촉 패치 입력**: 접촉 한 번을 점 하나가 아니라 접촉 위치 주변 국소 영역의 점 200개로 표현한다. 실제 촉각 센서가 한 점이 아닌 일정 면적의 패치를 관측하는 특성을 반영한 것으로, 점 하나보다 옥트리에 접촉 정보를 안정적으로 넣을 수 있다. 물체당 20번 접촉 → 4,000점이 입력 옥트리에 추가된다.
- **좌표로 결합**: 각 촉각 점은 그 좌표를 포함하는 voxel에 할당된다. 시각 feature와 촉각 점이 같은 3D 좌표계에서 자연스럽게 합쳐진다.
- **촉각 토큰**: 모델이 시각과 촉각을 구분하도록, 촉각 voxel에는 학습 가능한 단일 임베딩 벡터(촉각 토큰)를 똑같이 부여한다.
  - 시각 feature가 없는 voxel: 촉각 토큰만
  - 시각 feature가 있는 voxel: 시각 feature와 촉각 토큰을 결합
  - voxel 좌표가 촉각의 공간 위치를, 촉각 토큰이 모달리티 정보를 나눠 표현한다
- 이렇게 만든 시각-촉각 옥트리를 OctMAE 인코더-디코더에 그대로 넣어 단일 forward pass로 복원한다.

### 3-3. 손실 함수

OctMAE 원본 손실을 그대로 쓴다.

ℒ = ℒ_nrm + ℒ_SDF + Σ_{h ∈ {5,6,7,8,9}} ℒ_occ^h

| 항 | 내용 |
|---|---|
| ℒ_occ^h | LoD-h에서 예측한 occupancy의 binary cross-entropy. LoD-5~9 모든 해상도에서 계산해 계층적 표면 예측을 학습 |
| ℒ_SDF | LoD-9에서 예측한 signed distance의 L2 손실 |
| ℒ_nrm | LoD-9에서 예측한 표면 법선의 L2 손실 |

---

## 4. 결과

ShapeNet 6개 카테고리. 촉각 미사용 모델과 TactMAE를 무작위 초기화(scratch)와 OctMAE 사전 학습 가중치(finetune) 두 설정에서 학습해 비교. RTX 3090 4대.

| 모델 | CD (↓) | CD_vis (↓) | CD_occ (↓) | F1@10mm (↑) |
|---|---|---|---|---|
| OctMAE (사전 학습 그대로) | 5.347 | 3.260 | 8.280 | 0.8889 |
| 촉각 미사용 (scratch) | 5.087 | 1.753 | 9.655 | 0.8690 |
| TactMAE (scratch) | 2.529 | 1.513 | 4.537 | 0.9575 |
| 촉각 미사용 (finetune) | 4.165 | 1.711 | 7.349 | 0.8927 |
| **TactMAE (finetune)** | **2.270** | **1.477** | **4.381** | **0.9679** |

CD: Chamfer Distance (mm). CD_vis, CD_occ: 카메라에서 보이는 영역 / 가려진 영역만 따로 잰 값.

- 두 설정 모두 촉각을 넣으면 일관되게 좋아진다: CD 5.087 → 2.529 (scratch), 4.165 → 2.270 (finetune).
- 개선은 가려진 영역에서 특히 크다: finetune에서 CD_occ 7.349 → 4.381.
- 정성 결과에서도 베이스라인이 놓치는 가려진 뒷면과 컵 손잡이를 복원한다.

---

## 5. 확장: 능동 촉각 탐색

위 결과는 무작위로 고른 접촉점으로 얻은 것이다. 같은 접촉 횟수로 더 좋은 복원을 얻기 위해, 다음에 만질 위치를 고르는 정책을 학습한다 (`exploration/`).

| 요소 | 내용 |
|---|---|
| 환경 | 학습된 TactMAE (고정). 후보 접촉 20개 중 일부를 골라 입력 |
| 상태 | TactMAE MAE 인코더 latent (192차원, 현재 상태 + 촉각 없는 처음 상태), 사용한 접촉 마스크 |
| 행동 | 아직 안 쓴 접촉 하나 선택 |
| 보상 | 접촉 추가 후 Chamfer distance 감소량 |
| 정책 | DDQN (MLP Q-network, replay buffer, ε-greedy), 에피소드당 접촉 5번 |

향후 방향: 다수의 센서가 달린 촉각 장갑으로 여러 접촉을 동시에 얻어, 형상 복원과 조작을 함께 다루는 방향으로 확장.

---

## 6. 코드 구성

| 파일 | 내용 |
|---|---|
| `octmae/nets/model.py` | TactMAE (촉각 점 + 촉각 토큰을 입력 옥트리에 추가한 OctMAE) |
| `octmae/nets/model_latent_lod.py` | MAE를 LoD-6에서 수행하는 변형 |
| `octmae/nets/mae_encoder.py`, `mae_decoder.py`, `ocnn_blocks.py` | OctMAE 구성 요소 |
| `octmae/utils/misc.py` | 데이터 로딩, 학습 때 촉각 개수 0~20 무작위 샘플링 |
| `train.py`, `eval.py` | 학습 / 평가 (`--eval_touches`로 평가 접촉 수 지정) |
| `exploration/` | 능동 촉각 탐색 (DDQN) |
| `scripts/precompute_tactile.py`, `scripts/make_shards.py` | 데이터 생성 |
| `configs/` | 촉각 사용 여부 × scratch / finetune / LoD-6 |

---

## 참고문헌

- M. Comi et al., "TouchSDF: A DeepSDF approach for 3D shape reconstruction using vision-based tactile sensing," RA-L 2024.
- Y. Wang et al., "Touch2Shape: Touch-conditioned 3D diffusion for shape exploration and reconstruction," CVPR 2025.
- H. Zhang et al., "End-to-end diffusion-based 3D object reconstruction from robotic tactile sensing," RA-L 2026.
- S. Iwase et al., "Zero-shot multi-object scene completion," ECCV 2024.
- Y. Lin et al., "Tactile Gym 2.0," RA-L 2022.
- A. X. Chang et al., "ShapeNet," arXiv:1512.03012, 2015.
- M. Denninger et al., "BlenderProc," arXiv:1911.01911, 2019.
- N. F. Lepora, "Soft biomimetic optical tactile sensing with the TacTip: A review," IEEE Sensors Journal, 2021.
- S. Xie et al., "Aggregated residual transformations for deep neural networks (ResNeXt)," CVPR 2017.
