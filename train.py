"""
A single implementation of a logistic regression model for sentiment analysis.

Not the model answer; use only as a flow chart for reference.
"""

import nltk.lm
import numpy as np
import pandas as pd
import nltk
import itertools

from sklearn.linear_model import LogisticRegression

from typing import Iterable



def to_matrix(
    token_series: Iterable[list[str]],
    *,
    limit: int = 768,
) -> np.ndarray:
    volcabulary = nltk.lm.Vocabulary(itertools.chain(*dataset["tokens"]))
    all_words = sorted(
        nltk.lm.Vocabulary(itertools.chain(*dataset["tokens"])),
        key=lambda word: volcabulary[word],
        reverse=True,
    )[:limit]

    token_to_index = {token: index for index, token in enumerate(all_words)}

    matrix = np.zeros((len(token_series), len(all_words)), dtype=np.int32)

    for row_index, tokens in enumerate(token_series):
        for token in tokens:
            if token in token_to_index:
                matrix[row_index, token_to_index[token]] += 1

    return matrix

if __name__ == "__main__":
    dataset = pd.read_csv("data/twitter_training.csv").set_index("id")
    dataset["tokens"] = dataset["tweet"].apply(nltk.tokenize.casual_tokenize)

    print("Valid results are: \n", dataset["sentiment"].value_counts())
    labels = list(dataset["sentiment"].unique())

    X = to_matrix(dataset["tokens"])
    y = dataset["sentiment"].apply(labels.index)

    chosen_indices = np.random.choice(X.shape[0], X.shape[0] // 5, replace=False)
    test_mask = np.isin(np.arange(X.shape[0]), chosen_indices)
    X_test = X[test_mask]
    y_test = y[test_mask]

    X_train = X[~test_mask]
    y_train = y[~test_mask]

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Model fitted.")

    y_prediction = model.predict(X_test)
    accuracy = np.mean(y_prediction == y_test)

    print("Accuracy: ", accuracy)