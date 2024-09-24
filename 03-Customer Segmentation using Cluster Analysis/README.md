# Customer Segmentation using Cluster Analysis

This project performs customer segmentation using K-means clustering on customer data. It includes scripts for data preprocessing, optimal cluster number determination using the elbow method, and visualization of customer segments.

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
   cd 22CB903-Machine-Learning-MiniProjects/03-Customer\ Segmentation\ using\ Cluster\ Analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the customer segmentation script:
```
python customer_segmentation.py
```

## Project Structure

- `customer_segmentation.py`: Main script for data preprocessing, clustering, and visualization
- `Mall_Customers.csv`: Dataset containing customer information
- `requirements.txt`: List of Python dependencies

## Methodology

1. **Data Preprocessing**: 
   - Load the customer dataset using pandas
   - Inspect the dataset structure
   - Check for missing values
2. **Feature Selection**: 
   - Extract 'Annual Income (k$)' and 'Spending Score (1-100)' features
3. **Optimal Cluster Determination**:
   - Use the Elbow Method to find the optimal number of clusters
4. **K-means Clustering**: 
   - Apply K-means algorithm with the optimal number of clusters
5. **Visualization**: 
   - Plot the resulting clusters and their centroids

## Results

The script generates visualizations including:
- Elbow point graph for optimal cluster number determination
- Scatter plot of customer segments based on Annual Income and Spending Score

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
