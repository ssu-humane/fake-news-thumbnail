# How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image

**How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image**, Published in Findings of ACL workshop, CONSTRAINT 2022, [paper](https://arxiv.org/abs/2204.05533)

## 설명
+ fakenews_annotation.csv: 논문에서 사용한 두 개의 모델이 COVID-wo-faces 데이터셋의 가짜 뉴스 샘플에 대해 fakenews라고 예측한 기사들을 라벨링한 결과입니다. 라벨 1은 썸네일이 기사 제목을 잘 대표하지 않는 기사, 라벨 0은 썸네일이 기사 제목을 잘 대표하는 기사입니다.

## Train
```python
python train.py --image_path image \
                --train_path . \
                --val_path . \
                --learning_rate 0.0001 \
                --batch_size 128 \
                --num_epochs 10 \
                --save model_pt 
```
