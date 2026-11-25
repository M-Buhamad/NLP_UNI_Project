# News Topic Classification of Short News Headlines

NLP course project, Group 3, Section 7MA2 (Imam Abdulrahman Bin Faisal University)

We classify short news headlines from the AG News dataset into four topics: World, Sports, Business and Sci/Tech. The project compares two approaches:

1. **TF-IDF + Logistic Regression** (traditional approach)
2. **Sentence embeddings (all-MiniLM-L6-v2) + MLP classifier** (pretrained approach)

Both models are trained on the same training split and evaluated on the same official AG News test set (7,600 headlines).

## Results (test set)

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| --- | --- | --- | --- | --- |
| TF-IDF + LR | 0.8661 | 0.8656 | 0.8657 | 0.8661 |
| Embeddings + MLP | 0.8596 | 0.8595 | 0.8605 | 0.8596 |

## Project structure

```txt
NLP_Project/
├── data/                   train.csv and test.csv (downloaded automatically if missing)
├── models/                 trained models saved by the notebooks (.joblib)
├── notebooks/
│   ├── tfidf.ipynb         TF-IDF + Logistic Regression: training, testing, error analysis, Try It
│   └── embeddings.ipynb    Sentence embeddings + MLP: training, testing, error analysis, Try It
├── app/
│   ├── app.py              Streamlit app that compares both models on a new headline
│   └── run.py              Launcher that starts app.py with Streamlit
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Python 3.12 was used. Install the packages:

```bash
pip install -r requirements.txt
```

The exact package versions used for the reported results are printed in the second cell of each notebook.

## Data

The data is not stored in this repository because of its size. There are two options:

- **Automatic:** just run the notebook. If `data/train.csv` and `data/test.csv` are missing, the notebook downloads them from
  `https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/`
- **Manual:** download `train.csv` and `test.csv` from the link above and put them in the `data/` folder.

## How to run

1. Open `notebooks/tfidf.ipynb`, then Restart the kernel and Run All.
2. Open `notebooks/embeddings.ipynb`, then Restart the kernel and Run All.
   The first run downloads the `all-MiniLM-L6-v2` model (about 90 MB) and encodes all headlines, which can take 10 to 20 minutes on a CPU. The embeddings are saved in `data/`, so later runs are fast.
3. Run both notebooks in the same Python environment.

Each notebook trains its model, chooses the settings on the dev set, evaluates once on the test set and saves the model in `models/`.

## Try It

The last cell of each notebook loads the saved model from `models/` (no retraining) and classifies new headlines. Blank input shows a message asking for a headline.

### Streamlit app

There is also a small web app that shows the predictions of both models side by side, with the most likely topic, a confidence value for each model and a probability chart. If the two models predict different topics, the app tells you.

**Before you start:** run both notebooks first (see [How to run](#how-to-run)). The app only loads the saved models from `models/` (`tfidf_lr.joblib` and `emb_mlp.joblib`) and does not train anything. If they are missing, the app shows an error and stops.

**Run the app with `run.py`** (from the project root, with the same Python environment that has the packages installed):

```bash
python app/run.py
```

`run.py` checks that `app.py` exists and starts it with Streamlit (`python -m streamlit run app.py`), so you do not need to type the Streamlit command yourself. It uses the same Python interpreter that runs `run.py`, so the environment that has Streamlit installed is the one used.

Streamlit prints a local address in the terminal and usually opens it in your browser. If it does not open, go to `http://localhost:8501`. To stop the app, press `Ctrl+C` in the terminal.

**Using the app:**

1. Type a short news headline in the **Headline** box.
2. Click **Classify**. An empty box shows a warning asking you to write a headline.
3. Read the result: the predicted topic and confidence of each model, and a bar chart with the probability of each of the four topics.

The app loads the models once at start-up, so the first start takes a few seconds and later predictions are fast.

You can also start the app directly with Streamlit, without `run.py`:

```bash
streamlit run app/app.py
```

## Data split

| Split | Headlines | Use |
| --- | --- | --- |
| Train | 100,302 | training |
| Dev | 11,145 | choosing hyperparameters |
| Test | 7,600 | final evaluation only (official AG News test set) |

Train and dev come from the official training file after removing empty and duplicate headlines (stratified 90/10 split, `random_state=42`).

## Team

| Name | Student ID |
| --- | --- |
| Mohammad Buhamad (leader) | 2240005142 |
| Muhannad Almutlaq | 2240006060 |
| Mustafa Abokhder | 2240002889 |
