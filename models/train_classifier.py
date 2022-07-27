import sys
from string import punctuation

import pandas as pd

from sqlalchemy import create_engine
from joblib import dump, load
from pathlib import Path

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# from joblib.externals.loky import set_loky_pickler

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger", "omw-1.4", "stopwords"])
# set_loky_pickler("dill")
ENGLISH_STOPWORDS = stopwords.words("english")
# TODO: Add logger


class WordCount(BaseEstimator, TransformerMixin):
    '''
    Custom scikit-learn transformer to count the number of words in text.
    credit: https://github.com/rebeccaebarnes/DSND-Project-5/blob/master/scripts/train_classifier.py
    '''

    def word_count(self, text):
        table = text.maketrans(dict.fromkeys(punctuation))
        words = word_tokenize(text.lower().strip().translate(table))
        return len(words)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        count = pd.Series(x).apply(self.word_count)
        return pd.DataFrame(count)


class CharacterCount(BaseEstimator, TransformerMixin):
    '''
    Custom scikit-learn transformer to count the number of characters in text,
    including spaces and punctuation.
    credit: https://github.com/rebeccaebarnes/DSND-Project-5/blob/master/scripts/train_classifier.py
    '''

    def character_count(self, text):
        return len(text)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        count = pd.Series(x).apply(self.character_count)
        return pd.DataFrame(count)


class NounCount(BaseEstimator, TransformerMixin):
    '''
    Custom scikit-learn transformer to count the number of nouns in text after
    tokenization including removal of stop words, lemmatization of nouns and
    verbs, and stemming, using nltk's WordNetLemmatizer and PorterStemmer.
    credit: https://github.com/rebeccaebarnes/DSND-Project-5/blob/master/scripts/train_classifier.py
    '''

    def noun_count(self, text):
        count = 0
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            for _, tag in pos_tags:
                if tag in ['PRP', 'NN']:
                    count += 1

        return count

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        count = pd.Series(x).apply(self.noun_count)
        return pd.DataFrame(count)


class VerbCount(BaseEstimator, TransformerMixin):
    '''
    Custom scikit-learn transformer to count the number of nouns in text after
    tokenization using a custom "tokenize" function.
    credit: https://github.com/rebeccaebarnes/DSND-Project-5/blob/master/scripts/train_classifier.py
    '''

    def verb_count(self, text):
        count = 0
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            for _, tag in pos_tags:
                if tag in ['VB', 'VBP']:
                    count += 1

        return count

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        count = pd.Series(x).apply(self.verb_count)
        return pd.DataFrame(count)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) >= 1:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
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
    # https://stackoverflow.com/questions/44911539/pickle-picklingerror-args0-from-newobj-args-has-the-wrong-class-with-hado

    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    table = text.maketrans(dict.fromkeys(punctuation))

    tokens = word_tokenize(text.lower().strip().translate(table))
    tokens = [w for w in tokens if w not in ENGLISH_STOPWORDS]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        lemmed = lemmatizer.lemmatize(clean_tok, pos='v')
        stemmed = stemmer.stem(lemmed)
        clean_tokens.append(stemmed)

    return clean_tokens


def build_model():
    """
    Build Multioutput classifier Random Forest classifier model pipeline and
    run GridSearchCV to get optimised parameters
    :return: model pipeline
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ("text", TfidfVectorizer(
                tokenizer=tokenize, max_df=0.5,
                max_features=5000, ngram_range=(1, 2),
                use_idf=False)),
            ('starting_verb', StartingVerbExtractor()),
            ("word_count", WordCount()),
            ("character_count", CharacterCount()),
            ("noun_count", NounCount()),
            ("verb_count", VerbCount())
        ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier())),
    ])

    # specify parameters for grid search
    parameters = {
        "clf__estimator__n_estimators": [i for i in range(50, 350, 50)],
        "clf__estimator__max_depth": [10, 100, 200, 300, 400, 500],
        # "clf__estimator__max_features": ["auto", "sqrt", "log2"]
    }

    scorer = make_scorer(f1_score, average='micro')

    cv = GridSearchCV(pipeline, parameters, n_jobs=8, verbose=3, scoring=scorer)

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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
