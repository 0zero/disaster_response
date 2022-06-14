import sys
import pandas as pd

from sqlalchemy import create_engine
from joblib import dump, load
from pathlib import Path
import dill as pickle

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from joblib.externals.loky import set_loky_pickler

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger", "omw-1.4", "stopwords"])
set_loky_pickler("dill")

# TODO: Add logger


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        import nltk
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        import pandas as pd
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath: str):
    """
    Load sqlite database
    :param database_filepath: path to database file
    :return: tuple of data features, target variable and category names
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterTweets", engine)

    # TODO: drop child_alone column as it seems we have no instances of it being positive in
    #  dataset. Well do we want to do this?
    x = df.message.values
    y = df[df.columns[4:]].values
    category_names = df[df.columns[4:]].columns

    return x, y, category_names


def tokenize(text: str):
    """
    Process raw text input
    :param text: raw text input
    :return: cleaned text tokens
    """
    # https://www.mackelab.org/sbi/faq/question_03/
    import re
    import nltk

    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in nltk.corpus.stopwords.words("english")]
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Multioutput classifier Random Forest classifier model pipeline and
    run GridSearchCV to get optimised parameters
    :return: model pipeline
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier())),
    ])

    # specify parameters for grid search
    parameters = {
        # "features__text_pipeline__vect__ngram_range": ((1, 1), (1, 2)),
        "clf__estimator__n_estimators": [i for i in range(50, 225, 25)],
        "clf__estimator__max_depth": [10, 20, 30, 40, 50],
        "clf__estimator__max_features": ["auto", "sqrt", "log2"]
    }
    cv = GridSearchCV(pipeline, parameters, n_jobs=6, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print Classification report and accuracy of given model

    :param model: model to evaluate
    :param X_test: Test feature data
    :param Y_test: Test target data
    :param category_names: Names of categories representing targets
    :return: Precision, Recall, F1 score, and Accuracy of model is printed to terminal
    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        print(f"{col}\n", classification_report(Y_test[:, i], y_pred[:, i]))
        print(f"Accuracy: {accuracy_score(Y_test[:, i], y_pred[:, i])}")


def save_model(model, model_filepath):
    """
    Save trained model to path
    :param model: model to save
    :param model_filepath: path to save model to
    :return:
    """
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        # TODO: Add extra input so if model exists user can retrain and override model

        database_filepath, model_filepath = sys.argv[1:]

        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        if Path(model_filepath).exists():
            print(
                (
                    f"WARNING: The given model filepath '{model_filepath}' already exists.\n"
                    "So will just load the available model file instead of retraining."
                )
            )
            model = load(model_filepath)
        else:
            print("Building model...")
            model = build_model()

            print("Training model...")
            model.fit(X_train, Y_train)
            print(f"Best parameters:\n {model.best_params_}")

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        if Path(model_filepath).exists():
            print(
                (
                    f"WARNING: The given model filepath '{model_filepath}' already exists.\n"
                    "So will not be re-saving the model."
                )
            )
        else:
            print("Saving model...\n    MODEL: {}".format(model_filepath))
            save_model(model, model_filepath)
            print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database "
              "as the first argument and the filepath of the pickle file to "
              "save the model to as the second argument. \n\nExample: python "
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()
