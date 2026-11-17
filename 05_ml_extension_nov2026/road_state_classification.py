# -*- coding: utf-8 -*-
"""
Post-thesis extension — November 2026
======================================
Road-state classification (Open / Slow / Blocked) for post-earthquake networks.

Context
-------
In the Master's thesis, the T6 disrupted-network scenario was built by
classifying road segments manually from a seismic study (approx. 65% of the
network open, 25% slow, 10% blocked). This extension — November 2026, a first
step towards the PhD's prediction layer — trains a Random Forest classifier
that predicts a segment's state from its OpenStreetMap attributes, using the
thesis's own T6 labels: the blocked and slow segment sets exported from QGIS,
plus a random sample of the segments that remained usable.

Features are the OSM tags available for every segment: highway class,
surface, bridge/tunnel flags, one-way, lanes and maxspeed. The result is a
proof of concept: fragile rural classes (track, unclassified, unpaved)
dominate the blocked set, and the classifier recovers that signal.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
COLS = ["highway", "surface", "bridge", "tunnel", "oneway", "maxspeed", "lanes"]


def load_segments(path, label):
    df = pd.read_csv(path, low_memory=False,
                     usecols=lambda c: c in COLS)
    df["state"] = label
    return df


def main():
    blocked = load_segments(ROOT / "data" / "routes_blocked_t6.csv", "Blocked")
    slow = load_segments(ROOT / "data" / "routes_slow_t6.csv", "Slow")
    usable = load_segments(ROOT / "data" / "road_segments_usable_sample_t6.csv", "Open")
    data = pd.concat([blocked, slow, usable], ignore_index=True)
    print("Label counts:", data["state"].value_counts().to_dict())

    X = pd.get_dummies(data[COLS].fillna("missing").astype(str), dummy_na=False)
    y = data["state"]

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, X, y, cv=cv)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    print(f"\n5-fold CV accuracy : {acc:.3f}")
    print(f"5-fold CV macro F1 : {f1:.3f}")
    print("\n" + classification_report(y, preds, digits=3))

    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top features:\n", imp.head(8).to_string())
    pd.DataFrame({"accuracy": [acc], "macro_f1": [f1]}).to_csv(
        BASE / "road_state_model_results.csv", index=False)
    print("\nSaved: road_state_model_results.csv")


if __name__ == "__main__":
    main()
