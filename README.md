# 📘 **Emotion Classification for Chatbot Systems**

*A Classical Machine Learning Approach Using Naïve Bayes & Linear SVM*

---

## 📌 **Overview**

This project investigates whether classical machine learning models can accurately detect human emotions from text to improve emotional intelligence in chatbot systems. Using a large labelled dataset from Kaggle, the study benchmarks four text-classification models across six emotion categories:

**anger, fear, joy, love, sadness, and surprise**

Two feature extraction techniques (CountVectorizer and TF-IDF) and two classical models (Multinomial Naïve Bayes and Linear Support Vector Machine) were evaluated to identify the most effective combination.

---

## 📂 **Dataset**

**Source:** *Sentiment and Emotion Analysis Dataset* — Kaggle (Kushagra Agarwal, 2024)
**Size:** 422,746 text samples (reduced to 416,123 after removing duplicates)

* `text` — human-written sentence or short phrase
* `emotion` — labelled emotion category

The dataset contains a noticeable class imbalance, with **joy (33.8%)** and **sadness (29.0%)** dominating overall samples.

---

## 🧹 **Preprocessing Pipeline**

The preprocessing included:

* Lowercasing text
* Removing URLs, usernames, numbers, and punctuation
* Normalizing whitespace
* Encoding emotion labels → numeric values
* Splitting into training (75%) and testing (25%) using stratified sampling

A custom `cleantext()` function was used to prepare the corpus for feature extraction.

---

## 🔧 **Feature Extraction Methods**

### **1. CountVectorizer**

* Bag-of-words (unigram) representation
* `max_features=3000`

### **2. TF-IDF Vectorizer**

* Weighted term frequency
* `max_features=12000`
* `ngram_range=(1,2)` to include bigrams

---

## 🤖 **Models Implemented**

1. **Multinomial Naïve Bayes + CountVectorizer**
2. **Multinomial Naïve Bayes + TF-IDF**
3. **Linear SVM + CountVectorizer**
4. **Linear SVM + TF-IDF**

All models were evaluated using:

* Accuracy
* Weighted F1-score
* Macro AUC
* Micro AUC

---

## 📊 **Results Summary**

| Model              | Features        | Accuracy   | F1-Score   | Macro AUC  | Micro AUC  |
| ------------------ | --------------- | ---------- | ---------- | ---------- | ---------- |
| **Multinomial NB** | CountVectorizer | 0.8904     | 0.8900     | 0.9866     | 0.9885     |
| **Multinomial NB** | TF-IDF          | 0.8507     | 0.8425     | 0.9853     | 0.9826     |
| **Linear SVM**     | CountVectorizer | **0.9008** | **0.9006** | 0.9911     | 0.9925     |
| **Linear SVM**     | TF-IDF          | 0.9006     | 0.8997     | **0.9938** | **0.9945** |

### **Best Overall Model:**

✔ **Linear SVM + CountVectorizer**
Achieved the strongest balance across accuracy, generalization, and computational efficiency.

### Additional Insights:

* Naïve Bayes performs well with simple word counts but struggles with sparse TF-IDF features.
* SVM benefits from richer text representations but performs almost equally well with unigram counts.
* TF-IDF improves AUC but does not significantly improve SVM accuracy.

---

## 🧠 **Key Findings**

* Classical machine learning models can achieve **over 90% accuracy** in multi-class emotion classification.
* CountVectorizer (unigrams) is surprisingly competitive against more complex TF-IDF representations.
* SVM consistently outperforms Naïve Bayes but requires more computational effort.
* Naïve Bayes remains a strong baseline for lightweight or real-time applications.


