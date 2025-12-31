# DS102-DoAn
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
