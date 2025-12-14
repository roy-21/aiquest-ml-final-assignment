# Amazon Product Review Sentiment Analysis
This repository implements a sentiment analysis model for Amazon product reviews using ML techniques. It processes review data, trains classifiers to detect positive/negative, and visualizes results. Ideal for ML assignments!

### Project Overview

This project performs sentiment analysis on Amazon product reviews using a binary classification approach:

Positive: 1
Negative: 0

The pipeline includes text preprocessing, feature vectorization, model training, hyperparameter tuning, and performance evaluation. Multiple classical machine learning models were compared to identify the best-performing approach for short, review-based text data.

### Dataset

Source: Amazon Product Reviews
Nature: Textual reviews with sentiment labels
Observation: Dataset is positively skewed, requiring careful evaluation beyond accuracy

### Methodology

1️⃣ Text Preprocessing:- 
                        Lowercasing
                        Removing punctuation & stopwords
                        Handling contractions and typos
                        Lemmatization using NLTK

2️⃣ Feature Engineering:-
                        TF-IDF Vectorization
                        Word2Vec Embeddings

3️⃣ Models Trained:-
                    Logistic Regression (LR)
                    Random Forest (RF)
                    Support Vector Machine (SVM)
                    Naïve Bayes (NB)

4️⃣ Model Tuning:-
                  Grid Search for LR and SVM
                  F1-score used as the primary optimization metric

5️⃣ Evaluation Metrics:-
                        Accuracy
                        Precision
                        Recall
                        F1-score
                        Confusion Matrix

📊 Key Results

  Best Model: Naïve Bayes with TF-IDF
  Best F1-score: ~0.92
  Overall Accuracy: 90–95%
  Negative Class Recall: ~0.85

### Insights:-

TF-IDF outperformed Word2Vec by 3–5% F1-score
Grid Search improved LR & SVM performance by 1–2% F1-score
Accuracy alone was misleading due to class imbalance

### Challenges Faced:-

High dimensionality (5000+ TF-IDF features) slowed down SVM
Word2Vec required sufficient corpus size for quality embeddings
Class imbalance biased predictions toward positive sentiment
Cross-validation increased computational cost

### Key Learnings:-

TF-IDF is a strong and simple baseline for sentiment analysis
F1-score is essential for imbalanced classification problems
Pipelines improve reproducibility and workflow clarity
Model interpretability (LR coefficients) aids debugging
Ensemble models add robustness but increase computation time



### Tech Stack:-

Language: Python
Libraries: scikit-learn, NLTK, pandas, matplotlib, gensim
Vectorization: TF-IDF, Word2Vec

### Author

Sojib Chandra Roy
📎 LinkedIn: https://www.linkedin.com/in/roysojib/

If you find this project useful, consider giving it a star!
