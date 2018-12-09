import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

COLUMNS = (
    "ID",
    "TITLE",
    "URL",
    "PUBLISHER",
    "CATEGORY",
    "STORY",
    "HOSTNAME",
    "TIMESTAMP"
)


def read_dataset(infile="data/newsCorpora.csv", size=None):
    dataset = pd.read_table(infile, header=None, names=COLUMNS).dropna()
    target = dataset["CATEGORY"]
    return train_test_split(dataset.drop(columns=["CATEGORY"]), target)


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, records=False):
        self.columns = columns
        self.records = records

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.records:
            return X[self.columns].to_dict(orient="records")
        return X[self.columns]


def categorical(colname):
    return make_pipeline(
        PandasSelector(colname, records=True),
        DictVectorizer(),
    )


def build_model():
    model = make_pipeline(
        make_union(
            categorical(["PUBLISHER"]),
            # categorical(["HOSTNAME"]),
            make_pipeline(
                PandasSelector("TITLE"),
                CountVectorizer(stop_words="english",
                                ngram_range=(1, 2),
                                min_df=5)  # Regularize the model
            )
        ),
        # SGDClassifier(max_iter=10)
        LogisticRegression()
    )
    return model
