## [AmazonFood](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/code)


The original data is available from the [Kaggle website](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/code)

The preprocessed version of this data can be obtained from the original paper [Contrastive Learning for Sequential Recommendation](https://arxiv.org/pdf/2010.14395.pdf) via [ReChorus
](https://github.com/THUwangcy/ReChorus)

In this paper we use the preprocessed data from the CL4SRec baseline, however since the authors don't provide their split, we use our own split which is processed via the script below.


## Instructions

To process the data, use the following script, feeding the `amazon_food.df` file containing all the data: 
```
python data/replay_buffer_noreward.py
```