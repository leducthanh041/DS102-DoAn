# DS102 - FINAL PROJECT
## Acknowledgements

We would like to express our sincere gratitude to the instructors of this course for their guidance and support throughout the project:

- **Huỳnh Văn Tín (M.Sc.)** – Theory Instructor  
- **Nguyễn Văn Kiệt (Ph.D.)** – Theory Instructor  
- **Nguyễn Hiếu Nghĩa** – Practical Instructor  
- **Trần Quốc Khánh** – Practical Instructor  

## 1. Team Members

The project was conducted by the following team members, with clearly defined roles to ensure a consistent and high-quality annotation process.

| No. | Name | Role | Responsibilities |
|----|------|------|------------------|
| 1 | Duc Thanh Le - 23521441 | Project Lead - Reviewer | Overall project coordination, guideline design, quality control, inter-annotator disagreement resolution, final decision on ambiguous cases |
| 2 | Nhat Thanh Huynh - 23521440 | Annotator | Data annotation and error analysi, Data annotation according to the guideline |
| 3 | Thi Thanh Trang Bui - 23521625 | Annotator | Data annotation and error analysis |
| 4 | Tran Duy Truong Dinh - 23521688 | Annotator | Data annotation and error analysis |

## 2. Project Overview

This project focuses on building a high-quality labeled dataset for a **text classification task**, specifically aimed at **evaluating the sentiment polarity (Positive / Negative)** of news articles published on online media platforms.

The primary objective of the dataset is to support research and experimentation in sentiment analysis, opinion mining, and related natural language processing (NLP) tasks, with an emphasis on **Vietnamese news content**.

---

### 2.1 Data Source

- The raw data is collected from online news articles.
- Data is extracted from the following source:

[Vietnamnet](https://vietnamnet.vn/tin-tuc-24h)


- Each data sample corresponds to a single news item.
- The unit of annotation is **document-level**.

---

### 2.2 Annotation Guideline

- The final annotation guideline was carefully designed and refined through multiple iterations.
- The official and final version of the guideline is provided in the following file:

[Guideline](https://github.com/leducthanh041/DS102-DoAn/blob/main/GuildeLine-final.pdf)


- This guideline defines:
  - Sentiment labels
  - Decision rules
  - Priority rules
  - Edge cases and ambiguous scenarios

The guideline serves as the **authoritative reference** for all annotation and quality control activities in this project.

---

### 2.3 Annotation and Quality Control Process

To ensure annotation consistency and reliability, the project adopts a **multi-round annotator training and labeling workflow**.

The process includes:
- Progressive training of annotators
- Iterative refinement of the guideline
- Continuous quality control and disagreement analysis

Specifically:
- Annotators were trained across **7 rounds (Round 1 → Round 7)**.
- Each round is stored in a dedicated directory and includes:
  - Annotated samples
  - Disagreement cases
  - Review notes and corrections

This multi-stage design allows annotators to gradually align their understanding with the guideline and significantly improves annotation quality.

---

### 2.4 Overall Annotation Pipeline

The following figure illustrates the overall process of **annotator training, annotation execution, quality control, and final dataset construction**.

![Annotation Process](https://github.com/leducthanh041/DS102-DoAn/blob/main/Annotation-process.jpg)


The [final dataset](https://github.com/leducthanh041/DS102-DoAn/blob/main/final-round.json) is produced after all rounds are completed and all annotation conflicts are resolved by the reviewer.

## 3. Dataset Overview & Experiments

### 3.1 Dataset Overview

The dataset is constructed for a **sentiment classification task**, aiming to **evaluate the positive/negative polarity of online news articles** published on various media platforms.

- **Total size**: **2,444** samples  
- **Data split (7:1:2)**:
  - **Training set**: **1,710** samples (~70%)
  - **Development (Dev) set**: **244** samples (~10%)
  - **Test set**: **490** samples (~20%)

---

### 3.2 Text Preprocessing

A consistent text preprocessing pipeline is applied prior to model training, including the following steps:

1. Convert all text to **lowercase**
2. Remove **URLs**, **hashtags**, and **mentions**
3. Remove **digits** and **punctuation**
4. Perform **word segmentation** using **PyVi**: [trungtv/pyvi](https://github.com/trungtv/pyvi)
5. Remove **Vietnamese stopwords** using the stopword list from: [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords)
6. Remove **extra whitespace** (trimming and whitespace normalization)

---

### 3.3 Experimental Results

The following table summarizes the experimental results obtained on the final dataset, evaluated using **Accuracy** and **Macro-averaged F1 score (F1-macro)**.

| Category | Model | Pre-trained model | Accuracy (%) | F1-macro (%) |
|---|---|---|---:|---:|
| Machine Learning models | Logistic Regression | - | 70.41 | 70.57 |
| Machine Learning models | SVM | - | 70.61 | 70.67 |
| Machine Learning models | Naive Bayes | - | 71.43 | 71.12 |
| Machine Learning models | Decision Tree | - | 54.69 | 53.83 |
| Machine Learning models | KNN | - | 66.33 | 65.93 |
| Machine Learning models | XGBoost | - | 63.27 | 62.82 |
| Machine Learning models | Random Forest | - | 71.84 | 71.73 |
| Deep Learning models | Text CNN | - | 70.71 | 70.29 |
| Deep Learning models | GRU | - | 71.84 | 72.03 |
| Transformer models | BERT | bert-base-multilingual-cased | **75.92** | 75.86 |
| Transformer models | PhoBERT | phobert-base-v2 | **75.92** | **76.11** |

---

### 3.4 References (BibTeX)

```bibtex
@misc{pyvi,
  author       = {trungtv},
  title        = {PyVi: Python Vietnamese Core NLP Toolkit},
  howpublished = {\url{https://github.com/trungtv/pyvi}},
  note         = {Accessed 2025-12-31}
}

@misc{vietnamese_stopwords,
  author       = {stopwords},
  title        = {Vietnamese stopwords},
  howpublished = {\url{https://github.com/stopwords/vietnamese-stopwords}},
  note         = {Accessed 2025-12-31}
}

@misc{mbert,
  author       = {Google},
  title        = {bert-base-multilingual-cased},
  howpublished = {\url{https://huggingface.co/google-bert/bert-base-multilingual-cased}},
  note         = {Accessed 2025-12-31}
}

@misc{phobert_v2,
  author       = {VinAI Research},
  title        = {PhoBERT: phobert-base-v2},
  howpublished = {\url{https://huggingface.co/vinai/phobert-base-v2}},
  note         = {Accessed 2025-12-31}
}
