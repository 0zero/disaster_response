import pandas as pd
import plotly.graph_objs as go
from typing import Dict, Any


def plot_label_frequencies(df: pd.DataFrame) -> Dict[str, Any]:
    tmp_dict = {}
    for col in df.columns[4:]:
        tmp_dict.update({col: df[col].sum()})

    df_freq = pd.DataFrame.from_dict(tmp_dict, orient="index", columns=["frequency"])
    data = [go.Bar(x=list(df_freq.index), y=df_freq.frequency.values)]
    layout = {
        "title": "Occurrence of labels",
        "yaxis": {
            "title": "Counts"
        },
        "xaxis": {
            "title": {
                "text": "Message Categories",
                "standoff": 20
            },
            "automargin": "true",
        },
        "margin": {
            "l": 50,
            "r": 50,
            "b": 200,
            "t": 50,
            "pad": 4,
        }
    }
    return {"data": data, "layout": layout}


def plot_correlation_heatmap(df: pd.DataFrame) -> Dict[str, Any]:
    columns = df.columns[4:]
    correlation = df[columns].corr()
    data = [go.Heatmap(z=correlation, x=columns, y=columns, hoverongaps=False)]
    layout = {
        "title": "Correlation Coefficients for Categories",
        "xaxis_nticks": 36,
        "yaxis_nticks": 36,
        "width": 1200,
        "height": 1200,
        "margin": {
            "l": 200,
            "r": 50,
            "b": 200,
            "t": 50,
            "pad": 4,
        }
    }
    return {"data": data, "layout": layout}
