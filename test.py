import json
from datetime import datetime

from agents.questionPaperGeneratorAgent.questionPaperGenerator import QuestionPaperGenerator
text = """
Course: Programming Fundamentals and Data Structures

Course Outcomes:
CO1: Understand the basic concepts of programming, including variables, data types, and control structures.
CO2: Apply programming constructs such as loops, functions, and conditionals to solve computational problems.
CO3: Analyze and implement fundamental data structures to solve real-world problems efficiently.

Program Outcomes:
PO1: Engineering knowledge – Apply knowledge of mathematics and computing fundamentals.
PO2: Problem analysis – Identify, formulate, and analyze computational problems.
PO3: Design/development of solutions – Design and implement efficient algorithms.

Syllabus Content:
Unit 1: Introduction to programming, variables, data types, input/output operations.
Unit 2: Control structures – conditional statements, loops, and functions.
Unit 3: Arrays, strings, and basic operations.
Unit 4: Data structures – stacks, queues, linked lists.
Unit 5: Searching and sorting algorithms, time and space complexity.


"""
qpgen = QuestionPaperGenerator(collectionName="test_ai_collection_v1")
output = qpgen.demoQuestionpaperGenerator(text=text,filePath="")
print("final output")
print(output)

# Save output to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"question_paper_output_{timestamp}.json"

try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Output saved to: {output_filename}")
except Exception as e:
    # If output is not JSON serializable, save as text
    output_filename = f"question_paper_output_{timestamp}.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(str(output))
    print(f"\n✅ Output saved to: {output_filename}")




# from services.prompt.promptProcessor import dataProcessor
# processor = dataProcessor()
# content = {
#     "co": [
#         "CO2: Analyze algorithm efficiency using time and space complexity."
#     ],
#     "po": [
#         "PO3: Design/development of solutions"
#     ]
# }

# questions = [
#     "What is time complexity?",
#     "Define space complexity."
# ]

# verdict = (
#     "The questions are too theoretical and focus on definitions. "
#     "At least one Analyze-level question is required. "
#     "Include problem-based questions comparing algorithm efficiency."
# )

# memory = {
#     "pastToolCalls": [
#         {
#             "tool": "question_generator",
#             "questions": [
#                 "Explain the time complexity of binary search and compare it with linear search."
#             ]
#         }
#     ],
#     "stepReasoning": [
#         "Generated one Analyze-level comparison question, but coverage is still insufficient"
#     ]
# }

# output = processor.questionEvaluatorMainLoop(
#     content,
#     questions,
#     verdict,
#     memory
# )

# print(output)

# input = """
# 1️⃣ What is Machine Learning?

# Machine Learning (ML) is a field of AI where systems learn patterns from data instead of being explicitly programmed.
# The model discovers relationships → generalizes → makes predictions.

# Key Types

# Supervised Learning — learn from labeled data
# Examples: classification, regression

# Unsupervised Learning — learn from unlabeled data
# Examples: clustering, dimensionality reduction

# Semi-Supervised Learning — mix of labeled + unlabeled

# Reinforcement Learning — learn by interacting with environment

# Self-Supervised Learning — labels are generated from the data itself

# 2️⃣ Data in ML
# Dataset Components

# Features (X) — inputs

# Target/Label (y) — output

# Samples/Instances — rows

# Feature Types

# Numerical — continuous / discrete

# Categorical — nominal / ordinal

# Text

# Image/audio/video time-series

# Dataset Split

# Training set — learn parameters

# Validation set — tune hyperparameters

# Test set — evaluate final model

# Common split: 70% / 15% / 15% (or 80/20)

# 3️⃣ Data Pre-processing & Cleaning
# Handling Missing Values

# Delete rows (if few missing)

# Mean/median imputation

# Mode for categorical

# Advanced: KNN / iterative imputation

# Handling Outliers

# Z-score method

# IQR method

# Winsorization

# Log transform

# Feature Scaling

# Normalization (Min-Max)
# Good for neural networks

# Standardization (Z-score)
# Good for linear models & SVM

# Encoding Categorical Variables

# One-hot encoding

# Label encoding

# Target encoding

# Binary encoding

# Text Pre-processing

# Lowercasing

# Stopword removal

# Lemmatization/Stemming

# Tokenization

# Vectorization (TF-IDF, embeddings)

# 4️⃣ Feature Engineering
# Why?

# Improves model performance and interpretability.

# Techniques

# Polynomial features

# Interaction terms

# Domain-specific transformations

# Feature selection:

# Filter: correlation, chi-square

# Wrapper: RFE

# Embedded: LASSO, Tree importance

# Dimensionality reduction:

# PCA

# t-SNE

# UMAP

# 5️⃣ Supervised Learning Algorithms
# 📌 Regression (predict continuous values)

# Linear Regression

# y = mx + c

# Minimizes Mean Squared Error (MSE)

# Regularized Regression

# Ridge — L2 penalty

# Lasso — L1 penalty

# Elastic Net — L1 + L2

# Tree-based Regression

# Decision Trees

# Random Forest

# Gradient Boosting

# XGBoost / LightGBM / CatBoost

# 📌 Classification (predict categories)

# Logistic Regression
# Outputs probability using sigmoid.

# K-Nearest Neighbors (KNN)
# Instance-based learning.

# Naive Bayes
# Uses Bayes’ theorem + independence assumption.

# Decision Trees
# Splits based on information gain / Gini impurity.

# Ensembles

# Bagging — Random Forest

# Boosting — AdaBoost, GBM, XGBoost, LightGBM

# Stacking — meta-model combining others

# Support Vector Machines (SVM)
# Finds separating hyperplane, uses kernels.

# Neural Networks
# Multiple layers learn complex functions.

# 6️⃣ Unsupervised Learning
# Clustering

# K-Means

# Hierarchical Clustering

# DBSCAN

# Gaussian Mixture Models

# Dimensionality Reduction

# PCA

# t-SNE

# UMAP

# Autoencoders

# Association Rule Learning

# Apriori

# FP-Growth

# 7️⃣ Evaluation Metrics
# Regression Metrics

# MAE — Mean Absolute Error

# MSE — Mean Squared Error

# RMSE — Root MSE

# R² — Variance explained

# MAPE — Percentage error

# Classification Metrics

# Accuracy

# Precision

# Recall

# F1-Score

# ROC-AUC

# Log Loss

# Confusion Matrix
# 	Predicted +	Predicted –
# Actual +	TP	FN
# Actual –	FP	TN
# 8️⃣ Model Validation & Overfitting
# Bias-Variance Trade-off

# High bias → underfitting

# High variance → overfitting

# Prevent Overfitting

# Train-test split

# Cross-validation

# Regularization

# Early stopping

# Dropout (NNs)

# Simpler model

# Cross-Validation

# K-Fold (most common)

# Stratified K-Fold (classification)

# 9️⃣ Hyperparameter Tuning
# Methods

# Grid Search

# Random Search

# Bayesian Optimization

# Hyperband

# Genetic Algorithms

# Common Hyperparameters

# Learning rate

# Tree depth

# Number of trees

# Regularization strength

# Batch size (NNs)

# 🔟 Neural Networks & Deep Learning
# Basic Concepts

# Neuron = weighted sum + activation

# Layers:

# Input

# Hidden

# Output

# Activations

# ReLU

# Sigmoid

# Tanh

# Softmax

# Optimization

# Gradient Descent

# SGD

# Adam

# RMSProp

# Architectures

# CNNs → images

# RNN/LSTM/GRU → sequences

# Transformers → NLP

# Autoencoders → compression

# 1️⃣1️⃣ Reinforcement Learning
# Key Concepts

# Agent

# Environment

# Reward

# Policy

# Value function

# Algorithms

# Q-Learning

# Deep Q-Networks

# Policy Gradient

# 1️⃣2️⃣ ML Pipeline (End-to-End)

# Define problem

# Collect data

# Clean & preprocess

# EDA

# Feature engineering

# Split dataset

# Train model

# Tune hyperparameters

# Evaluate

# Deploy

# Monitor & retrain

# 1️⃣3️⃣ MLOps & Deployment
# Serving Methods

# REST API

# Batch inference

# Streaming inference

# Tools

# Flask / FastAPI

# Docker

# Kubernetes

# Airflow

# MLflow

# Kubeflow

# Monitoring

# Data drift

# Model drift

# Latency

# Accuracy decay

# 1️⃣4️⃣ Common ML Problems & Solutions
# Problem	Cause	Fix
# Overfitting	Complex model	Regularization, more data
# Underfitting	Too simple model	Add features, deeper model
# Data leakage	Using future info	Fix pipeline
# Imbalanced data	Skewed labels	SMOTE, class weights
# 1️⃣5️⃣ Ethics & Responsible AI

# Bias detection

# Fairness

# Transparency

# Privacy

# Explainability (SHAP, LIME)

# 1️⃣6️⃣ Important Concepts (Quick Reference)

# Curse of Dimensionality

# Cold start problem

# Bootstrapping

# Ensemble learning

# Gradient boosting vs bagging

# Cross-entropy loss

# Regularization (L1/L2)

# Feature importance

# ROC vs PR curves

# Stationarity in time-series

# Autocorrelation

# 1️⃣7️⃣ Time-Series ML (Brief)
# Techniques

# ARIMA / SARIMA

# Prophet

# LSTM

# Temporal CNN

# XGBoost on lag features

# Concepts

# Trend

# Seasonality

# Residuals

# Stationarity

# 1️⃣8️⃣ NLP Concepts

# Tokenization

# Stemming & Lemmatization

# TF-IDF

# Word2Vec & embeddings

# Transformers

# BERT/GPT

# Text classification

# Named Entity Recognition

# Sentiment analysis"""

# from agents.questionPaperGeneratorAgent.questionPaperGenerator import QuestionPaperGenerator
# qpgen = QuestionPaperGenerator(collectionName="test_ai_collection_v1")
# output = qpgen._recursiveChunker(input)
# base_metadata = {"CO": "CO1"}
# metadata_list = []
# for chunk in output:
#         meta = base_metadata.copy()
#         meta["text"] = chunk
#         metadata_list.append(meta)

# qpgen.ragSystem.process_and_store(chunks=output, metadata=metadata_list)
# content = {"co":"CO1"}
# generatedQuestions = ["what is ML","what is NLP"]
# verdict = "from notes add some other topics of Machine learning for four question"
# output = qpgen.mainQuestionPaperEvaluator(content=content,generatedQuestions=generatedQuestions,verdict=verdict)
# print(output)