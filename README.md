# How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image

**How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image**, Published in Findings of ACL workshop, CONSTRAINT 2022, [paper](https://arxiv.org/abs/2204.05533)

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
