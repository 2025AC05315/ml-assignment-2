# ML Assignment 2 - Breast Cancer Classification with Streamlit
# Machine Learning Assignment 2 - BITS

## a. Problem Statement

Breast cancer diagnosis is traditionally performed by manually examining
cell characteristics from a biopsy sample under a microscope. This project
builds and compares five machine learning classification models that predict
whether a breast tumor is **malignant** or **benign** based on numeric
features computed from a digitized image of a fine needle aspirate (FNA) of
a breast mass. The goal is to identify the model that best supports early,
accurate diagnosis, and to expose that model through an interactive
Streamlit web app for evaluation.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository / built into `scikit-learn`
  (`sklearn.datasets.load_breast_cancer`)
- **Instances:** 569 (meets the >=500 minimum)
- **Features:** 30 numeric features (meets the >=12 minimum), e.g. radius,
  texture, perimeter, area, smoothness, compactness, concavity, symmetry,
  and fractal dimension — each reported as mean, standard error, and
  "worst" value.
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** ~63% benign, ~37% malignant
- **Split used:** 80% train / 20% test (stratified, `random_state=42`)

## c. GitHub Repository Link

> **TODO:** Replace with your actual repo link after you push this project,
> e.g. `https://github.com/<your-username>/ml-assignment-2`

## d. Models Used

All 5 models were trained on the same 80/20 train-test split of the dataset
above, using standardized (scaled) features.

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                         | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                 | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)    | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

*(Values above come directly from `train_models.py` — re-run the script and
update this table if you change the random seed, split, or preprocessing.)*

### Observations

| ML Model Name              | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Best overall performer on this dataset — the classes are close to linearly separable in the scaled feature space, so a linear decision boundary generalizes very well. Highest accuracy, F1, and MCC of all five models. |
| Decision Tree                | Weakest performer. A single unpruned tree overfits the training data and is sensitive to small variations, which shows up as the lowest accuracy, AUC, and MCC. |
| kNN                          | Strong performer once features are standardized (kNN is distance-based, so scaling matters a lot). Comparable to Random Forest on most metrics. |
| Naive Bayes                  | Reasonable accuracy despite the strong (and technically violated) feature-independence assumption; the high AUC shows it ranks positives well even where its hard predictions are less precise. |
| Random Forest (Ensemble)     | Averaging many trees fixes the overfitting problem seen with the single Decision Tree, giving one of the highest AUC scores and matching kNN on Accuracy/F1/MCC. |
| **Overall Winner for your dataset?** | **Logistic Regression** — for this dataset, the simplest linear model achieved the best accuracy, precision/recall balance, and MCC, while also being the fastest to train and easiest to interpret. |

> **Note:** Re-run `train_models.py` and personalize this table if you swap
> in your own dataset choice, as encouraged by the assignment (using this
> exact dataset/observations verbatim across many student submissions is
> likely to be flagged in the anti-plagiarism check).

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit app
│-- train_models.py         # Trains all 5 models, saves them + test_data.csv
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split used for the Streamlit demo
│-- model/
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl
│   │-- feature_names.json
│   │-- metrics_comparison.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py     # trains models, writes model/*.pkl and test_data.csv
streamlit run app.py       # launches the interactive app
```

## How to Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App**, select your repo/branch, and set the main file to `app.py`.
4. Click **Deploy**.

## Live App

> **TODO:** Replace with your deployed Streamlit Community Cloud URL,
> e.g. `https://your-app-name.streamlit.app`
