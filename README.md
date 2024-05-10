# Description-based Stock Recommendation

This Jupyter Notebook demonstrates a simple approach to recommend stocks based on their descriptions using TF-IDF vectorization and cosine similarity.

## Dataset

The stock dataset is downloaded from a GitHub repository and contains information such as stock code, name, market segment, close price, transaction volume, market cap, shares outstanding, and a description of the company.

## Methodology

1. Load the dataset into a pandas DataFrame.
2. Compute the length of each stock's description and analyze the distribution.
3. Create a TF-IDF vectorizer to convert the descriptions into a matrix of TF-IDF features.
4. Calculate the cosine similarity between all pairs of stocks based on their TF-IDF vectors.
5. Define a function `get_recommendations` that takes a stock name as input and returns the top 10 most similar stocks based on their cosine similarity scores.

## Usage

To get recommendations for a specific stock, simply call the `get_recommendations` function with the desired stock name. The function will return the names of the top 10 most similar stocks.

## Dependencies

- pandas
- scikit-learn
