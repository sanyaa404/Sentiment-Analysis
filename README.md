## Hinglish Sentiment Analysis using NLP and Machine Learning

This project explores **sentiment analysis on Hinglish (Hindi–English code-mixed text)** using both traditional machine learning models and modern transformer architectures.

Social media in India frequently uses **code-mixed language**, where Hindi and English words appear in the same sentence, usually written in **Roman script**. Traditional sentiment analysis models trained on standard English text struggle to handle this noisy and mixed-language data.

This project evaluates multiple NLP techniques to identify sentiment in such data.

---

## Problem Statement

The rapid growth of social media platforms such as **Twitter, Reddit, Instagram, and YouTube comments** has created massive volumes of user-generated content.

However, this text is:

- Noisy and informal  
- Written using slang and abbreviations  
- Often code-mixed between Hindi and English  
- Frequently written in Romanized Hindi  

These characteristics make **standard NLP sentiment models unreliable**.

The goal of this project is to design and evaluate models that can **correctly classify sentiment in Hinglish text**.

---

## Objectives

The main objectives of the project are:

- Collect Hinglish sentiment datasets  
- Perform text preprocessing for code-mixed language  
- Extract features using TF-IDF  
- Train multiple machine learning models  
- Evaluate transformer-based multilingual models  
- Compare performance across approaches  

---

## Dataset

Two datasets were used in the project.

### Dataset 1
- Hinglish dataset  
- 2,766 samples  
- Contains mixed **Latin and Devanagari script**  

Labels:
- Positive  
- Negative  
- Neutral  

### Dataset 2
- Code-mixed multilingual dataset (Kaggle)  
- Originally 20,000 rows  
- Filtered to keep only **Hindi-English samples**

The datasets were combined and cleaned before training.

<img width="519" height="209" alt="Screenshot 2026-03-09 at 9 46 58 PM" src="https://github.com/user-attachments/assets/04fdbfdd-ff00-42ae-8716-1f366d54247a" />

---

## Preprocessing Pipeline

### General Cleaning

- Remove URLs  
- Remove punctuation  
- Remove HTML tags  
- Remove user mentions  
- Normalize whitespace  
- Convert emojis to sentiment words  
- Convert text to lowercase  
- Remove repeated characters

<img width="461" height="441" alt="Screenshot 2026-03-19 at 1 26 07 AM" src="https://github.com/user-attachments/assets/b967f2b8-d7ce-4fba-b5d8-91ae163d3481" />

### Code-Mixed Specific Handling

- Preserve mixed scripts  
- Keep Romanized Hindi words such as:
  - acha  
  - bakwas  
  - bekaar  
- Maintain word boundaries when scripts switch  

---

## Methodology

The project was divided into **two phases**.

---

## Phase 1 — Classical Machine Learning Models

Feature extraction was performed using **TF-IDF**.

Two feature representations were tested:

- Word-level TF-IDF  
- Character-level TF-IDF  

### Models Trained

- Logistic Regression  
- Support Vector Machine (SVM)  
- Multinomial Naive Bayes
  
<img width="534" height="262" alt="Screenshot 2026-03-09 at 9 47 57 PM" src="https://github.com/user-attachments/assets/dbeca95b-86c1-48af-abc2-c770574f3d62" />

---

## Phase 1 Results

### Word-Level TF-IDF

| Model | Accuracy |
|------|---------|
| Logistic Regression | 65.04% |
| SVM | 66.50% |
| Naive Bayes | 58.04% |

### Character-Level TF-IDF

| Model | Accuracy |
|------|---------|
| Logistic Regression | 70.08% |
| SVM | **73.82%** |
| Naive Bayes | 60.65% |

<img width="534" height="350" alt="Screenshot 2026-03-09 at 9 49 10 PM" src="https://github.com/user-attachments/assets/b3785c82-caea-4a5f-b39a-9905b56053aa" />


### Key Observation

Character-level features perform significantly better because they capture spelling variations like:

```
acha
accha
achha
```

These are common in Hinglish.

---

## Phase 2 — Transformer Models

We tested multilingual transformer models to see if they could better handle mixed-language text.

### Models Used

- **XLM-RoBERTa**
- **MuRIL**

Two configurations were tested:

1. Raw Hinglish text  
2. Devanagari text transliterated into Roman script  

---

## Transformer Results

| Model | Accuracy |
|------|---------|
| XLM-R (Raw Hinglish) | 60.25% |
| XLM-R (After Transliteration) | 37.60% |
| MuRIL | 60.03% |

### Important Finding

Transliteration **significantly reduces performance**.

This suggests that **natural Romanized Hinglish contains useful linguistic signals** that models rely on.

---

## Key Findings

### Best Model

SVM with **character-level TF-IDF** achieved the highest accuracy.

**Accuracy: 73.82%**

This also outperformed the previously reported baseline of **71.7%**.

---

### Major Insights

1. Hinglish sentiment analysis is significantly harder than English sentiment analysis.

2. Character-level TF-IDF works better because it captures:
   - spelling variations  
   - phonetic writing  
   - noisy text patterns  

3. Naive Bayes performs poorly due to noisy vocabulary distribution.

4. Transformer models do not perform significantly better due to **lack of Hinglish pretraining**.

5. Hinglish should be treated as its **own language variety**, not as English or Hindi.

---

## Technologies Used

- Python  
- Scikit-Learn  
- HuggingFace Transformers  
- Pandas  
- NumPy  
- TF-IDF  
- XLM-RoBERTa  
- MuRIL  

---

## Project Structure

```
project
│
├── data
│   ├── dataset1.csv
│   ├── dataset2.csv
│
├── notebooks
│   ├── preprocessing.ipynb
│   ├── training_models.ipynb
│
├── models
│
├── results
│   ├── confusion_matrix.png
│
└── README.md
```

---

## Future Work

Future improvements could include:

- Pretraining transformers specifically on **Romanized Hinglish**
- Using **custom tokenizers for code-mixed language**
- Leveraging **emoji-aware sentiment models**
- Collecting larger Hinglish datasets
