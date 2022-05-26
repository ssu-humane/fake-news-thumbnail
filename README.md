# Unrepresentative news thumbnail detection

## Task

Given a news title *T* and thumbnail image *I*, we want to detect news articles with unrepresentative thumbnail I.
The ground truth label is binary, indicating whether *I* represents *T* or not.

## Evaluation data

We obtained 259 news title-image pairs, of which label is manually annotated. We selected the annotated news articles by the prediction rank of [CLIP](https://openai.com/blog/clip/)-based models. We only provide news and image URLs with annotated labels for copyright issues. To obtain the original data, we recommend using [NewsPaper3k](https://newspaper.readthedocs.io/en/latest/), an open-sourced Python library.

This dataset was originally prepared for our task, but we think it could be further utilized for evaluation of models designed for vision-and-language tasks, such as image captioning, etc. Please refer to our [paper](https://arxiv.org/abs/2204.05533) for more detailed procedures on data selection and annotation.

## A baseline model

We implemented a baseline model that predicts the binary label on the thumbnail representativeness from the CLIP text and visual embeddings. You can train the model by using the command below. We don't provide the training dataset but you can make your own training dataset using the proposed method. You can find the details of dataset creation in our [paper](https://arxiv.org/abs/2204.05533).

```python
python train.py --image_path image \
                --train_path . \
                --val_path . \
                --learning_rate 0.0001 \
                --batch_size 128 \
                --num_epochs 10 \
                --save model_pt 
```

## Reference

All research outcomes (problem, model, and data) are presented as the ACL-22 workshop paper with the oral presentation at CONSTRAINT 2022. 
You are free to use our code and dataset, but please don't forget to cite our [work](https://arxiv.org/abs/2204.05533) if you publish an academic paper based on that. 

```bibtex
@inproceedings{choi-etal-2022-fake,
    title = "How does fake news use a thumbnail? {CLIP}-based Multimodal Detection on the Unrepresentative News Image",
    author = "Choi, Hyewon  and
      Yoon, Yejun  and
      Yoon, Seunghyun  and
      Park, Kunwoo",
    booktitle = "Proceedings of the Workshop on Combating Online Hostile Posts in Regional Languages during Emergency Situations",
    month = may,
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.constraint-1.10",
    pages = "86--94"
}
```


