import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from app.plotting import plot_label_frequencies, plot_correlation_heatmap
from models.train_classifier import tokenize, StartingVerbExtractor, WordCount, VerbCount, NounCount, CharacterCount

app = Flask(__name__)

# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("DisasterTweets", engine)

# load model
model = joblib.load("../models/classifierUpdatedFeatures.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
        {
            "data": [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {
                    "title": "Count"
                },
                "xaxis": {
                    "title": "Genre"
                }
            }
        }
    ]

    graphs.append(plot_label_frequencies(df))
    graphs.append(plot_correlation_heatmap(df))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        "go.html",
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
