# Comment Category Prediction

Machine-learning text-classification project. The task was to classify online comments into one of four categories (`0`–`3`) using text and structured metadata.

## Project Summary

- **Task:** 4-class text classification
- **Raw training data:** 198,000 comments
- **Test data:** 102,000 comments
- **Evaluation metric:** Macro F1
- **Final model:** LinearSVC
- **Best tuned C:** `0.1`
- **Best recorded 3-fold CV Macro F1:** `0.823`
- **Recorded Kaggle score:** **0.826**

The training data was highly imbalanced: Label 0 represented about 57.7% of the samples, while Label 3 represented only about 2.8%. Macro F1 was therefore used to give equal importance to performance across all classes.

## Approach

### 1. Exploratory Data Analysis

The notebook examines:

- class distribution and imbalance
- missing values
- comment-length patterns
- numeric-feature distributions and skewness
- categorical missingness
- feature correlations and outliers

One training row had a missing comment and was removed, leaving **197,999 usable training rows**.

## Dataset Analysis

The dataset is highly imbalanced, with Label 0 forming the majority class while Label 3 represents only a small fraction of the samples.

<p align="center">
  <img src="images/class_distribution.png" width="850" alt="Class Distribution">
</p>

<p align="center">
  <em>Class distribution showing the imbalance across the four target labels.</em>
</p>

### 2. Text Processing

Comments are:

- converted to lowercase
- stripped of URLs and special characters
- normalized for whitespace

### 3. Feature Engineering

The final pipeline combines:

- **60,000 word TF-IDF features** using 1–3 word n-grams
- **40,000 character TF-IDF features** using 3–5 character n-grams
- **20 numeric/engineered features**

This produces a final feature space of up to **100,020 features**.

Engineered features include comment length, word count, average word length, vote ratios, vote totals, emoticon counts, log-transformed skewed variables, weekend indicator, hour of day, and disability flag.

### 4. Models Compared

Three classifiers were evaluated:

- Logistic Regression
- LinearSVC
- Multinomial Naive Bayes

LinearSVC performed best in the recorded comparison and was selected for hyperparameter tuning.

### 5. Hyperparameter Tuning

`GridSearchCV` with 3-fold cross-validation tested:

```python
C = [0.05, 0.1, 0.3, 0.5]
```

The best recorded result was:

```text
Best C: 0.1
Best mean CV Macro F1: 0.823
```

The final LinearSVC was then trained on all 197,999 usable training rows.

## Model Evaluation

LinearSVC was selected as the final model after comparing multiple classical machine learning approaches.

The confusion matrix below shows the class-wise prediction behavior of the selected model.

<p align="center">
  <img src="images/confusion_matrix.png" width="550" alt="LinearSVC Confusion Matrix">
</p>

<p align="center">
  <em>Confusion matrix for the selected LinearSVC model.</em>
</p>

## Competition Result

The final recorded **Kaggle score was 0.826**.

This repository is a portfolio snapshot of the project as completed during the original competition.

## Repository Structure

```text
comment-category-prediction/
├── notebooks/
│   └── comment_category_prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Tech Stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Seaborn

## Notes on the Data

The competition dataset is **not included** in this repository. The notebook keeps the original Kaggle input paths used during the project.

## Key Takeaways

- Macro F1 is more informative than raw accuracy for this imbalanced multiclass task.
- Word and character TF-IDF provide complementary lexical and sub-word representations.
- Simple linear models can perform strongly on high-dimensional sparse text features.
- Regularization strength had a meaningful effect on LinearSVC performance.
- Minority-class performance remained the main challenge because of the severe class imbalance.