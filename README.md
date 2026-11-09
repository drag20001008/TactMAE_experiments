# TactMAE

단일 RGB-D 이미지와 촉각(tactile) 접촉점을 함께 입력받아 물체의 전체 3D 형상을 복원하는 모델.
[OctMAE](https://github.com/TRI-ML/OctMAE) (Iwase et al., ECCV 2024)를 기반으로, 촉각 입력과 능동 촉각 탐색(active touch exploration)을 추가했다.

> 이 레포는 OctMAE 코드를 수정한 것이다. 원본 OctMAE는 CC BY-NC 4.0 라이선스이며, 이 레포도 같은 라이선스를 따른다 ([라이선스](#라이선스)).

---

## OctMAE 대비 변경 사항

| 항목 | 내용 | 파일 |
|---|---|---|
| 촉각 입력 | 촉각 접촉점을 depth 점과 함께 입력 octree에 넣고, 촉각 점에는 RGB feature 대신 학습되는 `tactile_token`을 feature로 사용 | `octmae/nets/model.py`, `octmae/utils/misc.py` |
| 촉각 개수 변화 | 학습 때 촉각 개수를 `min_touches`~`max_touches`에서 무작위로 뽑음, 평가 때 `--eval_touches`로 고정 | `train.py`, `eval.py`, `configs/` |
| latent LoD 변경 | MAE latent를 LoD 6에서 처리하는 변형 모델 | `octmae/nets/model_latent_lod.py`, `train_latent_lod.py`, `eval_latent_lod.py` |
| 능동 촉각 탐색 | 학습된 TactMAE를 고정하고, 다음에 만질 위치를 고르는 DDQN 정책 학습 (보상: Chamfer distance 감소량) | `exploration/` |
| ShapeNet 데이터 | ShapeNet 렌더링 + 촉각 데이터를 OctMAE webdataset shard 형식으로 변환 | `scripts/make_shards.py`, `scripts/precompute_tactile.py` |
| 학습 안정화 | 빈 VOT(visible octree) 샘플 처리, grad inf/nan 스텝 건너뛰기, 빈 샘플 탐색 스크립트 | `octmae/nets/model.py`, `train.py`, `scripts/scan_vot_empty.py` |
| 시각화 | 입력, 촉각 점, 예측 결과 시각화 | `scripts/visualize_*.py` |

---

## 설치

```bash
git clone --recursive https://github.com/drag20001008/TactMAE.git
cd TactMAE
conda create -n octmae python=3.10 -y && conda activate octmae

# 1. PyTorch (CUDA 12.1)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 2. torch-scatter (ocnn 의존성)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 3. xformers
pip install -U xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121

# 4. voxel_occlusion_tester (submodule 빌드)
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
(cd submodules/voxel_occlusion_tester && python setup.py install)

# 5. requirements
pip install -r requirements.txt

# 6. shard 생성용 추가 패키지
pip install pysdf pycocotools
```

OctMAE 원본의 Docker 설치(`docker/`)도 그대로 남아 있다.

---

## 데이터 준비 (ShapeNet + 촉각)

ShapeNetCore 렌더링과 물체별 촉각 접촉 데이터가 필요하다 (데이터는 포함하지 않음).

```bash
# 1. 렌더링한 각 view의 카메라 좌표계로 촉각 점 변환 (view마다 tactile.npz 생성)
python scripts/precompute_tactile.py \
    --rendered_root {ShapeNetCoreV2Rendered} \
    --tactile_root {tactile_data} \
    --shapenet_root {ShapeNetCore}

# 2. webdataset shard 생성 (train / val / test 분할)
python scripts/make_shards.py \
    --rendered_root {ShapeNetCoreV2Rendered} \
    --shapenet_root {ShapeNetCore} \
    --output_dir {shard_dir}
```

생성한 shard 경로를 `configs/*.yaml`의 `train_dataset_url`, `val_dataset_url`, `test_dataset_url`에 넣는다.

---

## 학습과 평가

### 설정 파일

| config | 촉각 | 시작점 |
|---|---|---|
| `shapenet_tactile_finetune.yaml` | 사용 | OctMAE pretrained에서 fine-tune |
| `shapenet_tactile_scratch.yaml` | 사용 | 처음부터 학습 |
| `shapenet_no_tactile_finetune.yaml` | 없음 | OctMAE pretrained에서 fine-tune |
| `shapenet_no_tactile_scratch.yaml` | 없음 | 처음부터 학습 |
| `shapenet_tactile_lod6.yaml`, `shapenet_no_tactile_lod6.yaml` | 사용 / 없음 | latent LoD 6 모델 |

### 학습

```bash
# OctMAE pretrained 체크포인트 (fine-tune용)
mkdir -p checkpoints
wget https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/octmae/octmae.ckpt -P checkpoints/

# 촉각 fine-tune
python train.py --config configs/shapenet_tactile_finetune.yaml --run_name tactile_finetune --checkpoint checkpoints/octmae.ckpt

# 처음부터 학습
python train.py --config configs/shapenet_tactile_scratch.yaml --run_name tactile_scratch

# latent LoD 6
python train_latent_lod.py --config configs/shapenet_tactile_lod6.yaml --run_name tactile_lod6
```

### 평가

```bash
python eval.py --config configs/shapenet_tactile_finetune.yaml --checkpoint {ckpt} --eval_touches 5 --eval_name tactile_finetune_touch5

python eval_latent_lod.py --config configs/shapenet_tactile_lod6.yaml --checkpoint {ckpt} --eval_name tactile_lod6
```

### 능동 촉각 탐색

```bash
python exploration/train.py --checkpoint {tactmae_ckpt} --config configs/shapenet_tactile_scratch.yaml --exp_id my_run --budget 5
```

---

## 인용

이 코드는 OctMAE를 기반으로 한다. 사용 시 OctMAE 논문을 인용해 주세요.

```
@InProceedings{Iwase_ECCV_2024,
  author = {Iwase, Shun and, Liu, Katherine and Guizilini, Vitor and Gaidon, Adrien and Kitani, Kris and Ambruș, Rareș and Zakharov, Sergey},
  title = {Zero-Shot Multi-Object Scene Completion},
  booktitle = {ECCV},
  year = {2024}
}
```

---

## 라이선스

- 이 레포는 [OctMAE](https://github.com/TRI-ML/OctMAE)를 수정한 것으로, OctMAE와 같은 **[CC BY-NC 4.0](LICENSE.md)** 라이선스를 따른다 (비상업적 용도만 허용).
- 원본: OctMAE, Iwase et al., Toyota Research Institute 외. 변경 내용은 위 [OctMAE 대비 변경 사항](#octmae-대비-변경-사항) 참고.
- `submodules/voxel_occlusion_tester` ([TRI-ML/voxel_occlusion_tester](https://github.com/TRI-ML/voxel_occlusion_tester)): MIT License.
- ShapeNet, 촉각 데이터, OctMAE pretrained 체크포인트는 포함하지 않으며, 각자의 이용 약관을 따른다.
