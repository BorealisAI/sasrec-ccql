## [MovieLens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)


The original data is available from the [Kaggle website](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

The preprocessed version of this data can be obtained from the original paper [Contrastive Learning for Sequential Recommendation](https://arxiv.org/pdf/2010.14395.pdf) via [ReChorus
](https://github.com/THUwangcy/ReChorus)

In this paper we use the preprocessed data from the CL4SRec baseline, however since the authors don't provide their split, we use our own split which is processed via the script below.


## Instructions
The data needs to be split into the following `.df` files:
```
eval_buffer.df
movie_lens.df
train_replay_buffer.df
```

To process the data, use the following script, feeding the `.df` file containing all the data: 
```
python data/replay_buffer_noreward.py
```