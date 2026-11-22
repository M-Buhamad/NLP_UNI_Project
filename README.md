# News Topic Classification (AG News)

NLP Project - Group 3 - Section 7MA2

We classify short news headlines into 4 topics: World, Sports, Business, Sci/Tech.
We compare two methods:

1. TF-IDF (word + character n-grams) + Linear SVM
2. Sentence embeddings (all-MiniLM-L6-v2) + MLP (small neural network)

## Results (test set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + SVM | 0.8805 | 0.8802 |
| Embeddings + MLP | 0.8620 | 0.8619 |

## Project structure

```
NLP_Project/
├── data/               AG News train.csv and test.csv (+ saved embeddings)
├── models/             trained models (.joblib)
├── tfidf.ipynb         TF-IDF + SVM: training and testing
├── embeddings.ipynb    Embeddings + MLP: training and testing
├── app.py              Streamlit app (Try It)
└── requirements.txt
```

## Setup

```
pip install -r requirements.txt
```

## Run

Open the notebooks and run all cells:

- `tfidf.ipynb`
- `embeddings.ipynb` (encoding the headlines can take some time on CPU the first time)

Each notebook trains its model, tests it, and saves it in `models/`.

Then open the app:

```
streamlit run app.py
```

## Notes

- We use the headline (title) only, not the description.
- The source tag at the end of the headline like "(Reuters)" is removed.
- Train set is split 90/10 into train and dev. Dev is used to choose parameters.
- Test set is used only once at the end.
