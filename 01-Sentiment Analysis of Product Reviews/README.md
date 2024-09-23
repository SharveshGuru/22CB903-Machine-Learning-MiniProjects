# Sentiment Analysis of Product Reviews

This project performs sentiment analysis on product reviews scraped from Flipkart. It includes scripts for web scraping, data preprocessing, and sentiment analysis using NLTK and TextBlob.

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
   cd 22CB903-Machine-Learning-MiniProjects/01-Sentiment\ Analysis\ of\ Product\ Reviews
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the web scraping script to collect reviews:
   ```
   python extract_reviews.py
   ```

2. Perform sentiment analysis on the collected reviews:
   ```
   python sentimentanalysis.py
   ```

## Project Structure

- `extract_reviews.py`: Script for scraping reviews from Flipkart
- `sentimentanalysis.py`: Script for preprocessing and analyzing sentiment of reviews
- `reviews.csv`: CSV file containing the scraped reviews
- `requirements.txt`: List of Python dependencies

## Methodology

1. **Web Scraping**: Uses `requests` and `BeautifulSoup` to scrape reviews from Flipkart.
2. **Data Preprocessing**: Cleans the text by removing special characters and stopwords.
3. **Sentiment Analysis**: 
   - Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment scoring
   - Uses TextBlob for subjectivity scoring
4. **Classification**: Categorizes reviews as Positive, Negative, or Neutral based on the compound sentiment score.

## Results

The script generates a visualization of the sentiment distribution among the reviews and prints a summary of the results.

Example output:
```
Sentiment Distribution:
Positive    XXX
Neutral     XX
Negative    X
```

(Replace X's with actual numbers from your analysis)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
