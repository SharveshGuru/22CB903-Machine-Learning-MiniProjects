# Text Classification: Spam Detection

This project performs text classification to distinguish between spam and non-spam (ham) messages using two different feature extraction methods: TF-IDF and Word Embeddings (Word2Vec).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.x
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/SharveshGuru/22CB903-Machine-Learning-MiniProjects.git
   cd 22CB903-Machine-Learning-MiniProjects/02-Text\ Classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the text classification script:
```
python text_classification.py
```

## Project Structure

- `text_classification.py`: Main script for text preprocessing, feature extraction, and classification
- `requirements.txt`: List of Python dependencies
- `dataset.csv`: Dataset containing text messages and their labels (not included in the repository)

## Methodology

1. **Data Preprocessing**: 
   - Converts text to lowercase
   - Removes punctuation and special characters
   - Removes digits
   - Tokenizes the text

2. **Feature Extraction**:
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Word Embeddings (Word2Vec)

3. **Classification**:
   - Logistic Regression models trained on both TF-IDF and Word2Vec features

4. **Evaluation**:
   - Compares performance using metrics such as accuracy, precision, recall, and F1-score

## Results

The script outputs performance metrics for both TF-IDF and Word2Vec models, including:
- Accuracy
- Precision (for both ham and spam)
- Recall (for both ham and spam)
- F1-score (for both ham and spam)

Example output:
```
TF-IDF Model Performance:
              precision    recall  f1-score   support
         ham       0.97      1.00      0.98       865
        spam       1.00      0.77      0.87       150

Word2Vec Model Performance:
              precision    recall  f1-score   support
         ham       0.92      1.00      0.96       865
        spam       0.97      0.47      0.63       150
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
