# Enhanced Document Language Detection

This repository presents a comprehensive approach to language detection by comparing traditional and advanced methodologies. It leverages a Colab notebook to implement and evaluate three distinct models:

- **Embedding-Based Model** using [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- **Classical Naive Bayes Model** using [`imbesat-rizvi/language-identification`](https://github.com/imbesat-rizvi/language-identification)
- **Transformer-Based Model** using [`papluca/xlm-roberta-base-language-detection`](https://huggingface.co/papluca/xlm-roberta-base-language-detection)

The models are benchmarked on the [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification) dataset, evaluating their performance in terms of accuracy and F1 scores.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Language detection is a critical task in Natural Language Processing (NLP) with applications in content classification, preprocessing, and multilingual content management. This project compares three methodologies:

1. **Embedding-Based Approach**: Utilizes the `intfloat/multilingual-e5-large-instruct` model to generate embeddings and classify languages based on cosine similarity.
2. **Classical Naive Bayes**: Employs TF-IDF vectorization with a Multinomial Naive Bayes classifier for language prediction, implemented by `imbesat-rizvi/language-identification`.
3. **Transformer-Based Model**: Uses a fine-tuned `papluca/xlm-roberta-base-language-detection` transformer model for accurate language classification.

## Dataset

The project uses the [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification) dataset, which comprises text samples in 20 different languages. The dataset is divided into training, validation, and test sets.

## Installation

To replicate the results, install the necessary packages using the following command:

```bash
pip install datasets sentence-transformers numpy matplotlib seaborn pandas scikit-learn torch transformers tqdm
```

## Usage

The primary implementation is contained within a Colab notebook [Enhanced Document Language Detection.ipynb](https://colab.research.google.com/drive/1OIfFTvobq_M22cLcrFGsi6U0DkBkfkpE). You can navigate to the notebook available on the repository or run it directly on Google Colab. To run the notebook, make a copy and follow the step-by-step instructions.

## Models

### 1. Embedding-Based Model (`intfloat/multilingual-e5-large-instruct`)

- **Method**: Computes embeddings for texts and averages them per language. Classification is based on cosine similarity between test embeddings and average language embeddings.
- **Advantages**: High accuracy, good generalization, and scalability without the need for fine-tuning.

### 2. Classical Naive Bayes Model (`imbesat-rizvi/language-identification`)

- **Method**: Uses TF-IDF vectorization on character n-grams combined with a Multinomial Naive Bayes classifier.
- **Advantages**: Simplicity, efficiency, and good baseline performance.

### 3. XLM-RoBERTa Model (`papluca/xlm-roberta-base-language-detection`)

- **Method**: A fine-tuned transformer model specifically trained for language detection tasks.
- **Advantages**: High accuracy and robust multilingual capabilities.

## Evaluation

Each model is evaluated using the following metrics:

- **Accuracy**: Proportion of correctly classified samples.
- **F1 Score**: Weighted average of precision and recall.
- **Confusion Matrix**: Visual representation of classification performance across different languages.

## Results

| Model                       | Accuracy (%) | F1 Score |
|-----------------------------|--------------|----------|
| Embedding-Based Model (E5)  | 99.81        | 99.81    |
| Classical Naive Bayes Model | 99.22        | 99.22    |
| XLM-RoBERTa Model           | 99.60        | 99.60    |

The **Embedding-Based Model** achieves the highest accuracy, closely followed by the **XLM-RoBERTa Model** and the **Classical Naive Bayes Model**.

## Discussion

- **Embedding-Based Model**: Excels in accuracy and scalability but relies heavily on the quality of embeddings.
- **Classical Naive Bayes**: Offers simplicity and efficiency but may not capture complex linguistic nuances.
- **XLM-RoBERTa Model**: Balances high accuracy with robust multilingual support but requires more computational resources.

## Conclusion

The embedding-based approach using `intfloat/multilingual-e5-large-instruct` demonstrates superior performance in language detection tasks, offering a robust and scalable solution. While classical models provide a strong baseline, advanced transformer-based models offer enhanced accuracy suitable for more demanding applications.

## References

- **Language Identification Dataset**: [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)
- **Multilingual E5 Model**: [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- **XLM-RoBERTa Language Detection Model**: [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection)
- **Naive Bayes Language Identification**: [imbesat-rizvi/language-identification](https://github.com/imbesat-rizvi/language-identification)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or suggestions, please open an issue or contact [hassanmahmoudsd@example.com](mailto:hassanmahmoudsd@example.com).