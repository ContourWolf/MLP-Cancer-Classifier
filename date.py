"""
date.py - Modul pentru încărcarea, preprocesarea, normalizarea și împărțirea datelor.
Setul de date: Breast Cancer Wisconsin (Diagnostic).
"""

import os
import numpy as np
import pandas as pd


def incarca_si_preproceseaza(cale_fisier="data.csv"):
    """
    Încarcă fișierul CSV, elimină coloanele inutile ('id' și coloanele goale)
    și codifică eticheta 'diagnosis': Malign (M) = 1, Benign (B) = 0.
    
    Returnează:
        df: DataFrame-ul curățat
        X: matrice numpy cu cele 30 de trăsături numerice
        y: vector numpy cu etichetele (0 sau 1), formă (N, 1)
        nume_trasaturi: lista cu numele celor 30 de coloane de intrare
    """
    if not os.path.exists(cale_fisier):
        raise FileNotFoundError(
            f"Eroare: Fișierul '{cale_fisier}' nu a fost găsit în directorul curent! "
            "Asigură-te că data.csv este în același folder cu scriptul."
        )

    # Citire fișier CSV cu pandas
    df = pd.read_csv(cale_fisier)

    # 1. Eliminăm coloana 'id' dacă există (nu ajută la clasificare)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # 2. Eliminăm coloanele complet goale (cum ar fi 'Unnamed: 32')
    df = df.dropna(axis=1, how="all")
    coloane_unnamed = [col for col in df.columns if "unnamed" in col.lower()]
    if coloane_unnamed:
        df = df.drop(columns=coloane_unnamed)

    # 3. Codificăm coloana țintă 'diagnosis': M -> 1 (Malign), B -> 0 (Benign)
    df["diagnosis"] = df["diagnosis"].map({"M": 1.0, "B": 0.0})

    # Separăm trăsăturile de intrare (X) de etichetă (y)
    nume_trasaturi = [col for col in df.columns if col != "diagnosis"]
    X = df[nume_trasaturi].values.astype(np.float64)
    y = df["diagnosis"].values.reshape(-1, 1).astype(np.float64)

    return df, X, y, nume_trasaturi


def imparte_date(X, y, procent_tren=0.7, procent_val=0.2, procent_test=0.1, seed=42):
    """
    Împarte datele amestecate în 3 seturi: antrenare (70%), validare (20%), test (10%).
    Folosește un seed fix pentru reproductibilitate.
    
    Returnează:
        (X_tren, y_tren), (X_val, y_val), (X_test, y_test)
    """
    # Fixăm seed-ul pentru a obține mereu aceeași împărțire
    np.random.seed(seed)
    n_total = len(X)

    # Generăm indici amestecați aleator
    indici = np.random.permutation(n_total)

    # Calculăm numărul de exemple pentru fiecare set
    n_tren = int(n_total * procent_tren)
    n_val = int(n_total * procent_val)

    # Selectăm indicii pentru fiecare subset
    indici_tren = indici[:n_tren]
    indici_val = indici[n_tren : n_tren + n_val]
    indici_test = indici[n_tren + n_val :]

    X_tren, y_tren = X[indici_tren], y[indici_tren]
    X_val, y_val = X[indici_val], y[indici_val]
    X_test, y_test = X[indici_test], y[indici_test]

    return (X_tren, y_tren), (X_val, y_val), (X_test, y_test)


def calculeaza_parametri_normalizare(X_tren, tip="min-max"):
    """
    Calculează parametrii de scalare DOAR pe datele de antrenare
    (pentru a evita scurgerea de informație din validare/test).
    
    Metode:
        - 'min-max': scalare în intervalul [0, 1]
        - 'z-score': standardizare (medie 0, deviație standard 1)
    """
    if tip == "min-max":
        val_min = np.min(X_tren, axis=0)
        val_max = np.max(X_tren, axis=0)
        diferenta = val_max - val_min
        # Evităm împărțirea la zero pentru coloane constante
        diferenta[diferenta == 0.0] = 1.0

        return {
            "tip": "min-max",
            "minim": val_min.tolist(),
            "diferenta": diferenta.tolist(),
        }

    elif tip == "z-score":
        medie = np.mean(X_tren, axis=0)
        deviatie = np.std(X_tren, axis=0)
        # Evităm împărțirea la zero
        deviatie[deviatie == 0.0] = 1.0

        return {
            "tip": "z-score",
            "medie": medie.tolist(),
            "deviatie": deviatie.tolist(),
        }
    else:
        raise ValueError(f"Tip de normalizare necunoscut: {tip}. Folosește 'min-max' sau 'z-score'.")


def aplica_normalizare(X, parametri):
    """
    Aplică parametrii de normalizare calculați anterior pe un set de date (tren, val sau test).
    """
    tip = parametri.get("tip", "min-max")

    if tip == "min-max":
        val_min = np.array(parametri["minim"])
        diferenta = np.array(parametri["diferenta"])
        return (X - val_min) / diferenta

    elif tip == "z-score":
        medie = np.array(parametri["medie"])
        deviatie = np.array(parametri["deviatie"])
        return (X - medie) / deviatie

    else:
        raise ValueError(f"Tip de normalizare necunoscut: {tip}")


def pregateste_date_complete(cale_fisier="data.csv", tip_normalizare="min-max", seed=42):
    """
    Funcție ajutătoare care combină toți pașii:
    1. Încarcă și curăță datele
    2. Împarte în antrenare / validare / test (70% / 20% / 10%)
    3. Calculează parametrii de normalizare pe antrenare și îi aplică pe toate cele 3 seturi
    """
    df, X, y, nume_trasaturi = incarca_si_preproceseaza(cale_fisier)

    (X_tren, y_tren), (X_val, y_val), (X_test, y_test) = imparte_date(
        X, y, procent_tren=0.7, procent_val=0.2, procent_test=0.1, seed=seed
    )

    parametri_norm = calculeaza_parametri_normalizare(X_tren, tip=tip_normalizare)

    X_tren_norm = aplica_normalizare(X_tren, parametri_norm)
    X_val_norm = aplica_normalizare(X_val, parametri_norm)
    X_test_norm = aplica_normalizare(X_test, parametri_norm)

    return (
        (X_tren_norm, y_tren),
        (X_val_norm, y_val),
        (X_test_norm, y_test),
        parametri_norm,
        nume_trasaturi,
        df,
    )
