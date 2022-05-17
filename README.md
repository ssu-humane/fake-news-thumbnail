# How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image

**How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image**, Published in Findings of ACL workshop, CONSTRAINT 2022, [paper](https://arxiv.org/abs/2204.05533)

## 설명
+ fake_news_labeled_data.csv: COVID-wo-faces 데이터셋에서 가짜 뉴스 샘플에 대해 두 가지 모델로 라벨링한 결과입니다. CLIPScore 모델은 CLIPScore의 예측 점수에 (1 − similarity)을 사용한 값, CLIP-classifier 모델은 가짜 뉴스라고 예측한 확률 값을 기준으로 내림차순 정렬하여 200개씩 라벨링했습니다. 라벨 1은 썸네일이 기사 제목을 잘 대표하지 않는 기사, 라벨 0은 썸네일이 기사 제목을 잘 대표하는 기사입니다.

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
