"""
predict.py — the "brain" of our NIDS app.

It does one job: take raw network connections and score each one as
attack (1) or benign (0), using the RandomForest we trained in Phase 1.

Everything in here is objects talking to objects. The comments name the
TYPE of object at each step so you can see the OOP happening.
"""

import joblib          # joblib is a MODULE object (a toolbox of functions)
import pandas as pd    # pandas, nicknamed pd, is also a module object

# ---- Constants: plain data we reuse. (ALL_CAPS = "don't change me" by convention) ----
MODEL_PATH = "models/nids_rf_phase1.joblib"
CATEGORICAL = ["protocol_type", "service", "flag"]   # the 3 text columns to one-hot
DROP_IF_PRESENT = ["label", "difficulty"]            # not features; remove before scoring


def load_bundle(path=MODEL_PATH):
    """Load the saved bundle and hand back the two pieces we need."""
    bundle = joblib.load(path)      # joblib.load() is a METHOD on the joblib module.
                                    # It returns a DICT object -> that's `bundle`.
    model = bundle["model"]         # index into the dict -> a RandomForestClassifier OBJECT
    columns = bundle["columns"]     # index into the dict -> a LIST object (122 names)
    return model, columns           # hand both back to whoever called us


def score(df_raw, model, columns):
    """
    Score a table of raw connections.
    df_raw  = a DataFrame object of raw NSL-KDD rows (like sample_traffic.csv)
    model   = the RandomForestClassifier object from load_bundle()
    columns = the list of 122 feature names the model expects
    Returns a DataFrame with two new columns: prediction, attack_probability.
    """
    # .copy() is a METHOD on the DataFrame -> gives us our own copy to work on
    df = df_raw.copy()

    # 1) Drop non-feature columns IF they're present.
    #    .drop() is a DataFrame method; the [...] is a list we build on the fly.
    to_drop = [c for c in DROP_IF_PRESENT if c in df.columns]  # df.columns = attribute
    df = df.drop(columns=to_drop)

    # 2) One-hot encode the 3 text columns (same as your Task 3).
    encoded = pd.get_dummies(df, columns=CATEGORICAL, dtype=int)  # returns a new DataFrame

    # 3) Line the columns up EXACTLY with what the model was trained on.
    #    .reindex() is your Phase 1 Task 4 move: add missing cols as 0, drop extras, same order.
    X = encoded.reindex(columns=columns, fill_value=0)

    # 4) Ask the model. These are METHODS on the RandomForestClassifier object.
    predictions = model.predict(X)                 # array of 0/1
    probabilities = model.predict_proba(X)[:, 1]   # column [:,1] = probability of class 1 (attack)

    # 5) Attach results to a fresh copy of the ORIGINAL (so we keep label/service for display).
    result = df_raw.copy()
    result["prediction"] = predictions
    result["attack_probability"] = probabilities.round(3)
    return result


# This block only runs when you do `python app/predict.py` directly
# (not when another file imports this one). It's our quick self-test.
if __name__ == "__main__":
    model, columns = load_bundle()
    data = pd.read_csv("app/sample_traffic.csv")     # a DataFrame object
    scored = score(data, model, columns)

    n_attacks = (scored["prediction"] == 1).sum()    # sum() is a method on the Series
    print(f"Flagged {n_attacks} of {len(scored)} connections as attacks.")

    # Bonus: since the sample has a true `label`, check how accurate we were.
    if "label" in scored.columns:
        truth = (scored["label"] != "normal").astype(int)
        accuracy = (truth == scored["prediction"]).mean()
        print(f"Accuracy vs true labels: {accuracy:.1%}")
