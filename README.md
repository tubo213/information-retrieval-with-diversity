# information-retrieval-with-diversity

## Build Enviroment

1. [install rye](https://rye-up.com/guide/installation/)
2. build
```bash
rye sync
```

## How to train

1. download data
./data/rawにH&Mの[データ](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)置いてください

2. prepare data

```bash
rye run python bin/prepare_data.py
```

3. train

```
rye run python bin/train/item2vec.py
```

## Demo

```bash
rye run streamlit run app.py
```
[streamlit-app-2023-07-30-03-07-42.webm](https://github.com/tubo213/information-retrieval-with-diversity/assets/74698040/b5812f81-f584-4087-85cf-b7cc62c3c5cc)

## 実装済みのモデル

### Embedding Model
- Item2Vec
- (MatrixFactorization): なんか精度悪い

### Ranking Method
- Greedy: 関連度が高い順に推薦
- MRR: [The use of MMR, diversity-based reranking for reordering documents and producing summaries](https://dl.acm.org/doi/10.1145/290941.291025)
