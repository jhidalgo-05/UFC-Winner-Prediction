# UFC Fight Winner Prediction

## Overview

This project focuses on predicting the winner of UFC fights using machine learning classification models. The goal is to leverage historical fight data, fighter performance, recent form, and fighter-vs-fighter comparisons to generate predictions for upcoming bouts.

A major focus of the project is **feature engineering**. Rather than relying primarily on physical attributes such as height, reach, and weight, the project incorporates historical performance metrics to better capture a fighter's skill level and current form.

The project uses multiple approaches to represent fighter performance, including a composite **Fighter Score**, **rolling performance statistics**, and **exponential decay weighting** that prioritizes more recent fights.

---

# Objectives

* Predict UFC fight winners using machine learning classification models.
* Develop meaningful fighter-vs-fighter comparison features.
* Incorporate historical fighter performance and recent form.
* Create a composite Fighter Score to represent overall fighter performance.
* Use rolling statistics to capture recent performance trends.
* Apply exponential decay weighting to give greater importance to recent fights.
* Compare multiple machine learning models.
* Evaluate models using metrics appropriate for an imbalanced classification problem.
* Identify the features that contribute most to fight outcome predictions.

---

# Data

The dataset contains historical UFC fight information, including:

* Fighter statistics
* Significant striking statistics
* Takedown statistics
* Submission statistics
* Fight outcomes
* Fighter physical attributes
* Event dates

### Data Source

**Kaggle:** `maksbasher/ufc-complete-dataset-all-events-1996-2024`

---

# Methodology

## 1. Data Preprocessing

The data is prepared for machine learning by:

* Cleaning and organizing historical fight records.
* Handling missing values.
* Encoding categorical variables.
* Sorting fights chronologically.
* Creating fighter-vs-fighter comparison features.
* Ensuring that features are calculated using only information available **before the fight being predicted**.
* Preventing future fight information from leaking into the training data.

Avoiding data leakage is especially important for this project because the objective is to simulate how the model would perform when predicting an actual upcoming fight.

---

## 2. Fighter Score

A composite **Fighter Score** was created to provide a single measure of overall fighter performance.

The score incorporates several components:

* Career win rate
* Recent form
* Finish rate
* Experience score

The purpose of the Fighter Score is to combine both long-term performance and more recent results into a single feature.

The general concept is:

```text
Fighter Score =
    Career Win Rate
    + Recent Form
    + Finish Rate
    + Experience
```

The score is calculated using information available before each fight, allowing it to represent the fighter's historical ability at that point in their career.

For fight prediction, the model can use the difference between the two fighters:

```text
fighter_score_diff = red_fighter_score - blue_fighter_score
```

A positive value indicates an advantage for the Red fighter, while a negative value indicates an advantage for the Blue fighter.

---

## 3. Rolling Performance Statistics

Rolling statistics are used to capture a fighter's **recent performance** rather than relying only on career-level averages.

For each fighter, statistics are calculated from their previous fights while ensuring that the current fight is not included.

Examples of rolling fighter-vs-fighter features include:

* `str_acc_diff` — Striking Accuracy Difference
* `td_acc_diff` — Takedown Accuracy Difference
* `sub_att_diff` — Submission Attempts Difference
* `SLpM_total_diff` — Significant Strikes Landed per Minute Difference
* `str_def_total_diff` — Striking Defense Difference
* `td_def_total_diff` — Takedown Defense Difference

These features are designed to capture differences in recent fighting performance between the two competitors.

For example:

```text
str_acc_diff =
    Red Fighter Rolling Striking Accuracy
    - Blue Fighter Rolling Striking Accuracy
```

A positive value indicates that the Red fighter has the statistical advantage for that particular metric.

Rolling statistics help the model distinguish between a fighter's **current ability** and their entire career history.

---

## 4. Exponential Decay Weighting

The project also incorporates exponential decay weighting to give more importance to recent fights.

The general formula is:

```text
weight = exp(-λ * time_difference)
```

Where:

* `λ` controls how quickly older fights lose importance.
* `time_difference` represents the amount of time between the historical fight and the fight being predicted.

Therefore:

```text
Recent fight → Higher weight
Older fight  → Lower weight
```

This approach is intended to better capture changes in fighter performance over time.

For example, a fighter's performance from six months ago may provide more useful information about their current ability than a fight from eight years ago.

---

# Important Consideration: Exponential Decay

Although exponential decay can improve the representation of current fighter form, it introduces an important tradeoff.

Older fights receive less weight, meaning that historical model evaluation may not perfectly represent real-world predictive performance.

Potential issues include:

* Historical fights receiving substantially lower importance.
* Changes in fighter skill and competition level over time.
* Potential differences between historical and current UFC competition.
* Backtesting results not necessarily representing future performance.

Therefore, model performance should be interpreted carefully, particularly when comparing results across different periods.

---

# 5. Fighter-vs-Fighter Features

Instead of modeling each fighter independently, many features are represented as **differences between the Red and Blue fighters**.

Examples include:

```text
fighter_score_diff
str_acc_diff
td_acc_diff
sub_att_diff
SLpM_total_diff
str_def_total_diff
td_def_total_diff
```

The general structure is:

```text
feature_diff =
    Red Fighter Feature
    - Blue Fighter Feature
```

This allows the model to directly learn the relative advantage of one fighter over the other.

For example:

```text
Red SLpM: 4.8
Blue SLpM: 3.2

SLpM_total_diff = 1.6
```

The model can therefore learn whether larger advantages in specific performance categories are associated with a higher probability of winning.

---

# Models

Several machine learning classification models will be evaluated as part of the project.

The final models will be selected after comparing their performance using the evaluation metrics described below.

Potential models include:

* Support Vector Machine (SVM)
* Logistic Regression
* Gradient Boosting / XGBoost | LightGBM
* KNN

**Note:** The final model selection has not yet been determined. Models will be compared using the same cross-validation and evaluation framework before selecting the best-performing approach.

---

# Evaluation Metrics

Because the UFC dataset contains an **imbalanced target variable**, accuracy is not used as the primary evaluation metric.

Instead, the project focuses on:

### Precision

Measures how often the model is correct when it predicts a particular class.

```text
Precision = True Positives / (True Positives + False Positives)
```

High precision means that the model's positive predictions are generally reliable.

---

### Recall

Measures how effectively the model identifies actual positive cases.

```text
Recall = True Positives / (True Positives + False Negatives)
```

High recall means that the model misses fewer actual positive cases.

---

### F1 Score

The F1 score combines precision and recall into a single metric.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is particularly useful for this project because it balances the tradeoff between precision and recall.

### Why Accuracy Is Not Used

The dataset contains an imbalance in fight outcomes. Therefore, a model can achieve relatively high accuracy by favoring the majority class without necessarily learning meaningful patterns associated with the minority class.

For this reason, **precision, recall, and F1 score** are used as the primary evaluation metrics.

---

# Model Evaluation Strategy

Models will be evaluated using cross-validation to obtain a more reliable estimate of their generalization performance.

The same evaluation procedure will be applied across candidate models to ensure that their results are directly comparable.

The evaluation process will consider:

* Precision
* Recall
* F1 Score
* Cross-validation performance
* Confusion matrices
* Model consistency across folds

The final model will be selected based on its overall performance rather than a single metric.

---

# Key Considerations

### Avoiding Data Leakage

All historical and rolling features must be calculated using information that was available **before the fight occurred**.

The current fight's statistics or outcome must never be used when constructing its input features.

### Fighter Form Changes Over Time

A fighter's ability can change because of:

* Age
* Injuries
* Training camps
* Changes in fighting style
* Changes in competition level
* Time away from competition

Rolling statistics and exponential decay are therefore used to better represent current form.

### Physical Attributes

Physical attributes such as:

* Height
* Reach
* Weight
* Age

are included as potential predictors but are not assumed to be the primary determinants of fight outcomes.

The project places greater emphasis on **actual fighter performance and historical results**.

---

# Future Improvements

Potential future improvements include:

* Incorporating opponent quality into fighter ratings.
* Developing an Elo-based fighter rating system.
* Adding UFC-specific experience features.
* Incorporating significant strikes landed and absorbed.
* Adding takedown defense and submission rates.
* Improving the Fighter Score formulation.
* Testing additional rolling-window sizes.
* Optimizing the exponential decay parameter.
* Calibrating predicted probabilities.
* Testing ensemble models.
* Performing time-based backtesting to better simulate real-world predictions.
* Analyzing feature importance and model interpretability.

---

# Project Status

**Current Status:** Feature engineering and model development

The current phase focuses on developing meaningful pre-fight features and establishing a reliable evaluation framework. Machine learning models will be systematically compared before selecting the final predictive model.


5. Evaluation
Accuracy
ROC-AUC

Note: Traditional backtesting may be biased due to time-weighting.
