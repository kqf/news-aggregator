import click
from sklearn.metrics import classification_report
from model.model import build_model, read_dataset


@click.command()
@click.option("--path", type=click.Path(exists=True))
def main(path):
    X_tr, X_te, y_tr, y_te = read_dataset(path)
    model = build_model()
    model.fit(X_tr, y_tr)
    print("Train set", model.score(X_tr, y_tr))
    print("Test set ", model.score(X_te, y_te))
    print("Now the training set")
    print(classification_report(model.predict(X_tr), y_tr))
    print("Now the test set")
    print(classification_report(model.predict(X_te), y_te))


if __name__ == "__main__":
    main()
