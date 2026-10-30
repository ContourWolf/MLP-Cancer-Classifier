"""
gui.py - Interfața grafică pentru proiectul MLP - Clasificare Cancer Mamar
Folosește tkinter cu ttk.Notebook (funcționează perfect pe Windows).
Comunicarea între thread-ul de antrenare și GUI se face prin queue.Queue.
"""

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Importăm modulele noastre
from date import pregateste_date_complete, incarca_si_preproceseaza
from retea import MLP


# ─────────────────────────────────────────────────────────────────────────────
# Clasa principală a aplicației
# ─────────────────────────────────────────────────────────────────────────────
class AplicatieMLP:
    """
    Clasa principală care construiește și gestionează toată interfața grafică.
    Cele 3 tab-uri: Date, Antrenare, Test & Evaluare.
    """

    def __init__(self, root):
        # Fereastra principală
        self.root = root
        self.root.title("Proiect AI - Clasificare Cancer Mamar (MLP de la zero)")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        # ── Date și model (stare globală) ──────────────────────────────────
        self.date_tren = None      # (X_tren, y_tren)
        self.date_val  = None      # (X_val,  y_val)
        self.date_test = None      # (X_test, y_test)
        self.params_norm = None    # parametrii de normalizare
        self.df_brut = None        # DataFrame-ul original (pentru vizualizare)
        self.model = None          # instanța MLP curentă

        # ── Coadă pentru comunicarea thread antrenare → GUI ─────────────────
        self.coada_mesaje = queue.Queue()

        # ── Flag pentru oprirea antrenării ──────────────────────────────────
        self.oprire_solicitata = False

        # ── Construiește interfața ──────────────────────────────────────────
        self._construieste_interfata()

        # ── Încarcă datele la pornire ───────────────────────────────────────
        self._incarca_date()

    # ─────────────────────────────────────────────────────────────────────────
    # Construirea interfeței principale cu ttk.Notebook
    # ─────────────────────────────────────────────────────────────────────────
    def _construieste_interfata(self):
        """Construiește fereastra principală cu 3 tab-uri folosind ttk.Notebook."""

        # Aplicăm tema 'clam' - funcționează perfect pe Windows, aspect curat
        stil = ttk.Style()
        stil.theme_use("clam")

        # Stiluri globale
        stil.configure("TFrame",      background="#f0f0f0")
        stil.configure("TLabel",      background="#f0f0f0", font=("Segoe UI", 13))
        stil.configure("TButton",     font=("Segoe UI", 13), padding=[8, 6])
        stil.configure("TRadiobutton",background="#f0f0f0", font=("Segoe UI", 13))
        stil.configure("TLabelframe", background="#f0f0f0")
        stil.configure("TLabelframe.Label", background="#f0f0f0", font=("Segoe UI", 13, "bold"))
        stil.configure("TNotebook",   background="#f0f0f0")
        stil.configure("TNotebook.Tab", font=("Segoe UI", 13, "bold"), padding=[16, 7])

        # Font global folosit explicit pe fiecare widget (override sigur pe Windows)
        self.FONT_NORMAL = ("Segoe UI", 13)
        self.FONT_BOLD   = ("Segoe UI", 13, "bold")
        self.FONT_TITLU  = ("Segoe UI", 15, "bold")
        self.FONT_MARE   = ("Segoe UI", 16, "bold")

        # Notebook-ul principal (tab-urile)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # Creăm cele 3 frame-uri pentru tab-uri
        self.tab_date      = ttk.Frame(self.notebook)
        self.tab_antrenare = ttk.Frame(self.notebook)
        self.tab_test      = ttk.Frame(self.notebook)

        # Adăugăm tab-urile în notebook
        self.notebook.add(self.tab_date,      text="  📊 1. Date  ")
        self.notebook.add(self.tab_antrenare, text="  🧠 2. Antrenare  ")
        self.notebook.add(self.tab_test,      text="  🎯 3. Test & Evaluare  ")

        # Construim conținutul fiecărui tab
        self._construieste_tab_date()
        self._construieste_tab_antrenare()
        self._construieste_tab_test()

    # =========================================================================
    # TAB 1: DATE
    # =========================================================================
    def _construieste_tab_date(self):
        """Construiește tab-ul 'Date': statistici, normalizare, grafice."""

        # Layout grid responsive: coloana 0 = stânga (25%), coloana 1 = dreapta (75%)
        self.tab_date.columnconfigure(0, weight=1)
        self.tab_date.columnconfigure(1, weight=3)
        self.tab_date.rowconfigure(0, weight=1)

        F  = self.FONT_NORMAL
        FB = self.FONT_BOLD
        FT = self.FONT_TITLU

        # ── Panoul din stânga ────────────────────────────────────────────────
        pan_stanga = ttk.Frame(self.tab_date)
        pan_stanga.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ttk.Label(pan_stanga, text="Setul de date", font=FT).pack(pady=(10, 6))

        # Statistici
        cadru_stat = ttk.LabelFrame(pan_stanga, text="Statistici")
        cadru_stat.pack(fill="x", pady=4)

        self.lbl_total     = ttk.Label(cadru_stat, text="Total: -",         font=F)
        self.lbl_benigni   = ttk.Label(cadru_stat, text="Benigni (B): -",   font=F)
        self.lbl_maligni   = ttk.Label(cadru_stat, text="Maligni (M): -",   font=F)
        self.lbl_trasaturi = ttk.Label(cadru_stat, text="Trăsături: 30",    font=F)
        for w in [self.lbl_total, self.lbl_benigni, self.lbl_maligni, self.lbl_trasaturi]:
            w.pack(anchor="w", padx=10, pady=4)

        # Normalizare
        cadru_norm = ttk.LabelFrame(pan_stanga, text="Normalizare")
        cadru_norm.pack(fill="x", pady=8)

        self.var_normalizare = tk.StringVar(value="min-max")
        ttk.Radiobutton(cadru_norm, text="Min-Max  [0, 1]",
                        variable=self.var_normalizare, value="min-max",
                        style="TRadiobutton").pack(anchor="w", padx=10, pady=4)
        ttk.Radiobutton(cadru_norm, text="Z-Score  (μ=0, σ=1)",
                        variable=self.var_normalizare, value="z-score",
                        style="TRadiobutton").pack(anchor="w", padx=10, pady=4)

        ttk.Button(pan_stanga, text="🔄  Reîncarcă date.csv",
                   command=self._incarca_date).pack(fill="x", pady=10)

        # Împărțire
        cadru_imp = ttk.LabelFrame(pan_stanga, text="Împărțire (70 / 20 / 10)")
        cadru_imp.pack(fill="x", pady=4)

        self.lbl_tren = ttk.Label(cadru_imp, text="Antrenare: -", font=F)
        self.lbl_val  = ttk.Label(cadru_imp, text="Validare:  -", font=F)
        self.lbl_test = ttk.Label(cadru_imp, text="Test:      -", font=F)
        for w in [self.lbl_tren, self.lbl_val, self.lbl_test]:
            w.pack(anchor="w", padx=10, pady=4)

        # ── Panoul din dreapta (grafice) ─────────────────────────────────────
        pan_dreapta = ttk.Frame(self.tab_date)
        pan_dreapta.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        pan_dreapta.rowconfigure(0, weight=1)
        pan_dreapta.columnconfigure(0, weight=1)

        self.fig_date, self.axe_date = plt.subplots(2, 2, figsize=(8, 5))
        self.fig_date.patch.set_facecolor("#f0f0f0")
        self.fig_date.subplots_adjust(hspace=0.45, wspace=0.35)

        self.canvas_date = FigureCanvasTkAgg(self.fig_date, master=pan_dreapta)
        self.canvas_date.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # =========================================================================
    # TAB 2: ANTRENARE
    # =========================================================================
    def _construieste_tab_antrenare(self):
        """Construiește tab-ul 'Antrenare': hiperparametri, butoane, grafic timp real."""

        self.tab_antrenare.columnconfigure(0, weight=1)
        self.tab_antrenare.columnconfigure(1, weight=3)
        self.tab_antrenare.rowconfigure(0, weight=1)

        F  = self.FONT_NORMAL
        FB = self.FONT_BOLD
        FT = self.FONT_TITLU

        # ── Panoul din stânga ────────────────────────────────────────────────
        pan_stanga = ttk.Frame(self.tab_antrenare)
        pan_stanga.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ttk.Label(pan_stanga, text="Hiperparametri", font=FT).pack(pady=(10, 6))

        cadru_hp = ttk.LabelFrame(pan_stanga, text="Configurare rețea")
        cadru_hp.pack(fill="x", pady=4)

        def rand_param(parinte, eticheta, widget_fn):
            """Rând cu etichetă (font mare) și câmp de intrare."""
            f = ttk.Frame(parinte)
            f.pack(fill="x", padx=8, pady=5)
            ttk.Label(f, text=eticheta, font=F, anchor="w").pack(side="left",
                                                                  fill="x", expand=True)
            w = widget_fn(f)
            w.pack(side="right")
            return w

        self.entry_straturi = rand_param(
            cadru_hp, "Straturi ascunse:",
            lambda p: ttk.Entry(p, width=8, font=F))
        self.entry_straturi.insert(0, "8")

        self.combo_activare = rand_param(
            cadru_hp, "Activare ascunsă:",
            lambda p: ttk.Combobox(p, values=["sigmoid", "tanh"],
                                   state="readonly", width=9, font=F))
        self.combo_activare.set("sigmoid")

        self.entry_epoci = rand_param(
            cadru_hp, "Max epoci:",
            lambda p: ttk.Entry(p, width=8, font=F))
        self.entry_epoci.insert(0, "100")

        self.entry_rata = rand_param(
            cadru_hp, "Rata de învățare:",
            lambda p: ttk.Entry(p, width=8, font=F))
        self.entry_rata.insert(0, "0.1")

        self.entry_eroare = rand_param(
            cadru_hp, "Eroare admisă:",
            lambda p: ttk.Entry(p, width=8, font=F))
        self.entry_eroare.insert(0, "0.001")

        self.combo_init = rand_param(
            cadru_hp, "Init. ponderi:",
            lambda p: ttk.Combobox(p, values=["Xavier", "Aleatoare mici"],
                                   state="readonly", width=9, font=F))
        self.combo_init.set("Xavier")

        # Butoane antrenare
        cadru_butoane = ttk.LabelFrame(pan_stanga, text="Control antrenare")
        cadru_butoane.pack(fill="x", pady=8)

        self.btn_antreneaza = ttk.Button(cadru_butoane, text="▶  Antrenează",
                                         command=self._porneste_antrenarea)
        self.btn_antreneaza.pack(fill="x", padx=8, pady=5)

        self.btn_opreste = ttk.Button(cadru_butoane, text="⏹  Oprește",
                                      command=self._opreste_antrenarea,
                                      state="disabled")
        self.btn_opreste.pack(fill="x", padx=8, pady=(0, 5))

        # Stare antrenare
        cadru_stare = ttk.LabelFrame(pan_stanga, text="Stare antrenare")
        cadru_stare.pack(fill="x", pady=4)

        self.lbl_stare  = ttk.Label(cadru_stare, text="Stare: inactiv",    font=F)
        self.lbl_epoca  = ttk.Label(cadru_stare, text="Epocă: -",          font=F)
        self.lbl_mse_tr = ttk.Label(cadru_stare, text="MSE antrenare: -",  font=F)
        self.lbl_mse_v  = ttk.Label(cadru_stare, text="MSE validare:  -",  font=F)
        for w in [self.lbl_stare, self.lbl_epoca, self.lbl_mse_tr, self.lbl_mse_v]:
            w.pack(anchor="w", padx=10, pady=3)

        # Salvare / Încărcare
        cadru_model = ttk.LabelFrame(pan_stanga, text="Model (JSON)")
        cadru_model.pack(fill="x", pady=8)

        ttk.Button(cadru_model, text="💾  Salvează model...",
                   command=self._salveaza_model).pack(fill="x", padx=8, pady=5)
        ttk.Button(cadru_model, text="📂  Încarcă model...",
                   command=self._incarca_model).pack(fill="x", padx=8, pady=(0, 5))

        # ── Panoul din dreapta (grafic MSE) ──────────────────────────────────
        pan_dreapta = ttk.Frame(self.tab_antrenare)
        pan_dreapta.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        pan_dreapta.rowconfigure(1, weight=1)
        pan_dreapta.columnconfigure(0, weight=1)

        ttk.Label(pan_dreapta, text="Evoluția erorii MSE pe epoci",
                  font=FT).grid(row=0, column=0, pady=(6, 0))

        self.fig_tren, self.ax_tren = plt.subplots(figsize=(7, 5))
        self.fig_tren.patch.set_facecolor("#f0f0f0")
        self._reseteaza_grafic_antrenare()

        self.canvas_tren = FigureCanvasTkAgg(self.fig_tren, master=pan_dreapta)
        self.canvas_tren.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    # =========================================================================
    # TAB 3: TEST & EVALUARE
    # =========================================================================
    def _construieste_tab_test(self):
        """Construiește tab-ul 'Test & Evaluare': metrici și grafice de performanță."""

        self.tab_test.columnconfigure(0, weight=1)
        self.tab_test.columnconfigure(1, weight=3)
        self.tab_test.rowconfigure(0, weight=1)

        F  = self.FONT_NORMAL
        FB = self.FONT_BOLD
        FT = self.FONT_TITLU
        FM = self.FONT_MARE

        # ── Panoul din stânga ────────────────────────────────────────────────
        pan_stanga = ttk.Frame(self.tab_test)
        pan_stanga.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ttk.Label(pan_stanga, text="Metrici performanță", font=FT).pack(pady=(10, 6))

        cadru_met = ttk.LabelFrame(pan_stanga, text="Rezultate pe setul de test")
        cadru_met.pack(fill="x", pady=4)

        self.lbl_acuratete = ttk.Label(cadru_met, text="Acuratețe:  -", font=FM)
        self.lbl_precizie  = ttk.Label(cadru_met, text="Precizie:   -", font=F)
        self.lbl_recall    = ttk.Label(cadru_met, text="Recall:     -", font=F)
        self.lbl_f1        = ttk.Label(cadru_met, text="F1-Score:   -", font=F)
        for w in [self.lbl_acuratete, self.lbl_precizie, self.lbl_recall, self.lbl_f1]:
            w.pack(anchor="w", padx=10, pady=5)

        ttk.Separator(cadru_met, orient="horizontal").pack(fill="x", padx=8, pady=4)

        cadru_mc = ttk.LabelFrame(pan_stanga, text="Matrice de confuzie")
        cadru_mc.pack(fill="x", pady=8)

        self.lbl_tp = ttk.Label(cadru_mc, text="TP (Malign corect): -", font=F)
        self.lbl_tn = ttk.Label(cadru_mc, text="TN (Benign corect): -", font=F)
        self.lbl_fp = ttk.Label(cadru_mc, text="FP (Fals pozitiv):  -", font=F)
        self.lbl_fn = ttk.Label(cadru_mc, text="FN (Fals negativ):  -", font=F)
        for w in [self.lbl_tp, self.lbl_tn, self.lbl_fp, self.lbl_fn]:
            w.pack(anchor="w", padx=10, pady=4)

        ttk.Button(pan_stanga, text="🔍  Evaluează pe setul de test",
                   command=self._evalueaza_model).pack(fill="x", pady=12)

        # ── Panoul din dreapta (grafice) ──────────────────────────────────────
        pan_dreapta = ttk.Frame(self.tab_test)
        pan_dreapta.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        pan_dreapta.rowconfigure(0, weight=1)
        pan_dreapta.columnconfigure(0, weight=1)

        self.fig_test, self.axe_test = plt.subplots(1, 2, figsize=(8, 5))
        self.fig_test.patch.set_facecolor("#f0f0f0")
        self.fig_test.subplots_adjust(wspace=0.4)

        self.canvas_test = FigureCanvasTkAgg(self.fig_test, master=pan_dreapta)
        self.canvas_test.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # =========================================================================
    # Logica: Încărcare date
    # =========================================================================
    def _incarca_date(self):
        """Încarcă și normalizează datele; actualizează statisticile și graficele."""
        tip_norm = self.var_normalizare.get()

        # Căutăm data.csv relativ la locația scriptului
        cale_script = os.path.dirname(os.path.abspath(__file__))
        cale_csv = os.path.join(cale_script, "data.csv")

        try:
            rezultat = pregateste_date_complete(
                cale_fisier=cale_csv,
                tip_normalizare=tip_norm,
                seed=42
            )
            (X_tren, y_tren), (X_val, y_val), (X_test, y_test), params_norm, _, df_brut = rezultat

            self.date_tren   = (X_tren, y_tren)
            self.date_val    = (X_val,  y_val)
            self.date_test   = (X_test, y_test)
            self.params_norm = params_norm
            self.df_brut     = df_brut

            # Actualizăm statisticile în panoul din stânga
            n_total    = len(df_brut)
            n_benigni  = int((df_brut["diagnosis"] == 0).sum())
            n_maligni  = int((df_brut["diagnosis"] == 1).sum())

            self.lbl_total.config(   text=f"Total: {n_total}")
            self.lbl_benigni.config( text=f"Benigni (B): {n_benigni}")
            self.lbl_maligni.config( text=f"Maligni (M): {n_maligni}")
            self.lbl_tren.config(    text=f"Antrenare: {len(X_tren)}")
            self.lbl_val.config(     text=f"Validare:  {len(X_val)}")
            self.lbl_test.config(    text=f"Test:      {len(X_test)}")

            # Desenăm graficele de vizualizare
            self._deseneaza_grafice_date(df_brut)

        except FileNotFoundError as e:
            messagebox.showerror("Eroare", str(e))
        except Exception as e:
            messagebox.showerror("Eroare la încărcarea datelor", str(e))

    def _deseneaza_grafice_date(self, df):
        """Desenează 4 grafice: distribuția claselor + 3 histograme comparative."""
        # Curățăm graficele vechi
        for ax in self.axe_date.flat:
            ax.clear()

        culori = {"Benign": "#4CAF50", "Malign": "#F44336"}

        df_b = df[df["diagnosis"] == 0]
        df_m = df[df["diagnosis"] == 1]

        # Grafic 1: Distribuția claselor (bar chart)
        ax0 = self.axe_date[0, 0]
        ax0.bar(["Benign", "Malign"],
                [len(df_b), len(df_m)],
                color=[culori["Benign"], culori["Malign"]],
                edgecolor="white", width=0.5)
        ax0.set_title("Distribuția claselor", fontsize=9, fontweight="bold")
        ax0.set_ylabel("Nr. exemple")
        for i, v in enumerate([len(df_b), len(df_m)]):
            ax0.text(i, v + 3, str(v), ha="center", fontsize=9)

        # Graficele 2, 3, 4: histograme pentru 3 trăsături cheie
        trasaturi_de_afisat = [
            ("radius_mean",   "Raza medie"),
            ("texture_mean",  "Textura medie"),
            ("area_mean",     "Aria medie"),
        ]
        pozitii = [(0, 1), (1, 0), (1, 1)]

        for (coloana, titlu), (r, c) in zip(trasaturi_de_afisat, pozitii):
            ax = self.axe_date[r, c]
            if coloana in df.columns:
                ax.hist(df_b[coloana], bins=20, alpha=0.6, color=culori["Benign"],
                        label="Benign", edgecolor="white")
                ax.hist(df_m[coloana], bins=20, alpha=0.6, color=culori["Malign"],
                        label="Malign", edgecolor="white")
            ax.set_title(titlu, fontsize=9, fontweight="bold")
            ax.set_ylabel("Frecvență")
            ax.legend(fontsize=7)

        self.fig_date.tight_layout()
        self.canvas_date.draw()

    # =========================================================================
    # Logica: Antrenare (thread separat + queue)
    # =========================================================================
    def _porneste_antrenarea(self):
        """Citește hiperparametrii din GUI și pornește antrenarea într-un thread separat."""
        if self.date_tren is None:
            messagebox.showwarning("Date lipsă", "Datele nu sunt încărcate! Apasă 'Reîncarcă date.csv'.")
            return

        # ── Citim și validăm hiperparametrii ────────────────────────────────
        try:
            # Straturi ascunse: ex. "8" → [8] sau "8,4" → [8, 4]
            text_straturi = self.entry_straturi.get().strip()
            straturi_ascunse = [int(x.strip()) for x in text_straturi.split(",") if x.strip()]
            if not straturi_ascunse:
                raise ValueError("Introdu cel puțin un strat ascuns.")

            activare    = self.combo_activare.get()
            max_epoci   = int(self.entry_epoci.get())
            rata_inv    = float(self.entry_rata.get())
            eroare_adm  = float(self.entry_eroare.get())
            init_ales   = self.combo_init.get()
            metoda_init = "xavier" if init_ales == "Xavier" else "aleatoare"

        except ValueError as e:
            messagebox.showerror("Parametru invalid", str(e))
            return

        # ── Construim arhitectura rețelei ────────────────────────────────────
        # Intrare: 30 neuroni (trăsăturile tumorii), Ieșire: 1 neuron (malign/benign)
        dimensiuni = [30] + straturi_ascunse + [1]

        # ── Creăm rețeaua și resetăm graficul ───────────────────────────────
        self.model = MLP(
            dimensiuni_straturi=dimensiuni,
            activare_ascunsa=activare,
            metoda_init=metoda_init
        )

        self._reseteaza_grafic_antrenare()

        # Listele pentru datele graficului în timp real
        self.epoci_grafic   = []
        self.erori_tren_grafic = []
        self.erori_val_grafic  = []

        # ── Actualizăm butoanele ─────────────────────────────────────────────
        self.btn_antreneaza.config(state="disabled")
        self.btn_opreste.config(state="normal")
        self.lbl_stare.config(text="Stare: ▶ antrenare...")
        self.oprire_solicitata = False

        # ── Pornim thread-ul de antrenare ────────────────────────────────────
        X_tren, y_tren = self.date_tren
        X_val,  y_val  = self.date_val

        def callback_epoca(epoca, mse_tren, mse_val):
            """Trimite datele epocii în coada de mesaje (apelat din thread)."""
            self.coada_mesaje.put({
                "tip":     "epoca",
                "epoca":   epoca,
                "mse_tren": mse_tren,
                "mse_val":  mse_val
            })

        def thread_antrenare():
            """Funcția care rulează antrenarea în fundal."""
            try:
                self.model.antreneaza(
                    X_tren, y_tren.flatten(),
                    X_val,  y_val.flatten(),
                    max_epoci=max_epoci,
                    rata_invatare=rata_inv,
                    eroare_admisa=eroare_adm,
                    callback_epoca=callback_epoca,
                    oprire_solicitata=lambda: self.oprire_solicitata
                )
            except Exception as e:
                self.coada_mesaje.put({"tip": "eroare", "mesaj": str(e)})
            finally:
                self.coada_mesaje.put({"tip": "terminat"})

        fir = threading.Thread(target=thread_antrenare, daemon=True)
        fir.start()

        # Pornim polling-ul cozii de mesaje (la fiecare 50 ms)
        self.root.after(50, self._proceseaza_coada)

    def _proceseaza_coada(self):
        """
        Procesează mesajele din coadă și actualizează GUI-ul.
        Apelată periodic de root.after() — rămâne în main thread (thread-safe).
        """
        try:
            while True:
                mesaj = self.coada_mesaje.get_nowait()

                if mesaj["tip"] == "epoca":
                    epoca   = mesaj["epoca"]
                    mse_tr  = mesaj["mse_tren"]
                    mse_v   = mesaj["mse_val"]

                    # Actualizăm etichetele de stare
                    self.lbl_epoca.config( text=f"Epocă: {epoca}")
                    self.lbl_mse_tr.config(text=f"MSE antrenare: {mse_tr:.5f}")
                    self.lbl_mse_v.config( text=f"MSE validare:  {mse_v:.5f}")

                    # Adăugăm punctele la grafic
                    self.epoci_grafic.append(epoca)
                    self.erori_tren_grafic.append(mse_tr)
                    self.erori_val_grafic.append(mse_v)

                    # Redesenăm graficul (la fiecare 5 epoci sau la ultima)
                    if epoca % 5 == 0 or epoca == 1:
                        self._actualizeaza_grafic_antrenare()

                elif mesaj["tip"] == "terminat":
                    # Antrenarea s-a finalizat
                    self._actualizeaza_grafic_antrenare()
                    self.btn_antreneaza.config(state="normal")
                    self.btn_opreste.config(state="disabled")
                    self.lbl_stare.config(text="Stare: ✓ finalizat")

                    # Evaluăm automat pe setul de test
                    self._evalueaza_model()
                    return  # Oprim polling-ul

                elif mesaj["tip"] == "eroare":
                    messagebox.showerror("Eroare antrenare", mesaj["mesaj"])
                    self.btn_antreneaza.config(state="normal")
                    self.btn_opreste.config(state="disabled")
                    self.lbl_stare.config(text="Stare: ✗ eroare")
                    return

        except queue.Empty:
            pass

        # Continuăm polling-ul dacă antrenarea mai rulează
        self.root.after(50, self._proceseaza_coada)

    def _opreste_antrenarea(self):
        """Setează flagul de oprire — thread-ul de antrenare va vedea și se va opri."""
        self.oprire_solicitata = True
        self.lbl_stare.config(text="Stare: ⏹ oprire solicitată...")
        self.btn_opreste.config(state="disabled")

    def _reseteaza_grafic_antrenare(self):
        """Curăță graficul de antrenare și îl pregătește pentru o nouă sesiune."""
        self.ax_tren.clear()
        self.ax_tren.set_title("Eroarea MSE pe epoci", fontsize=10, fontweight="bold")
        self.ax_tren.set_xlabel("Epocă")
        self.ax_tren.set_ylabel("MSE")
        self.ax_tren.grid(True, linestyle="--", alpha=0.5)
        self.ax_tren.legend(["Antrenare", "Validare"])
        if hasattr(self, "canvas_tren"):
            self.canvas_tren.draw()

    def _actualizeaza_grafic_antrenare(self):
        """Redesenează graficul MSE cu datele curente (antrenare=albastru, validare=portocaliu)."""
        self.ax_tren.clear()
        self.ax_tren.plot(self.epoci_grafic, self.erori_tren_grafic,
                          color="#1976D2", linewidth=1.5, label="Antrenare")
        self.ax_tren.plot(self.epoci_grafic, self.erori_val_grafic,
                          color="#FF9800", linewidth=1.5, linestyle="--", label="Validare")
        self.ax_tren.set_title("Eroarea MSE pe epoci", fontsize=10, fontweight="bold")
        self.ax_tren.set_xlabel("Epocă")
        self.ax_tren.set_ylabel("MSE")
        self.ax_tren.legend(fontsize=8)
        self.ax_tren.grid(True, linestyle="--", alpha=0.5)
        self.canvas_tren.draw()

    # =========================================================================
    # Logica: Evaluare pe setul de test
    # =========================================================================
    def _evalueaza_model(self):
        """Evaluează modelul pe setul de test și afișează metricile și graficele."""
        if self.model is None:
            messagebox.showwarning("Model lipsă", "Antrenează sau încarcă un model mai întâi!")
            return
        if self.date_test is None:
            messagebox.showwarning("Date lipsă", "Datele de test nu sunt disponibile!")
            return

        X_test, y_test = self.date_test
        rezultate = self.model.evalueaza(X_test, y_test.flatten())

        # Actualizăm etichetele cu metricile
        self.lbl_acuratete.config(text=f"Acuratețe:  {rezultate['acuratete']*100:.2f}%")
        self.lbl_precizie.config( text=f"Precizie:   {rezultate['precizie']*100:.2f}%")
        self.lbl_recall.config(   text=f"Recall:     {rezultate['recall']*100:.2f}%")
        self.lbl_f1.config(       text=f"F1-Score:   {rezultate['f1']*100:.2f}%")
        self.lbl_tp.config(text=f"TP (Malign corect): {rezultate['tp']}")
        self.lbl_tn.config(text=f"TN (Benign corect): {rezultate['tn']}")
        self.lbl_fp.config(text=f"FP (Fals pozitiv):  {rezultate['fp']}")
        self.lbl_fn.config(text=f"FN (Fals negativ):  {rezultate['fn']}")

        # Desenăm graficele de evaluare
        self._deseneaza_grafice_test(rezultate)

        # Comutăm automat pe tab-ul Test
        self.notebook.select(self.tab_test)

    def _deseneaza_grafice_test(self, rezultate):
        """Desenează matricea de confuzie și graficul cu metrici."""
        for ax in self.axe_test:
            ax.clear()

        # ── Grafic stânga: Matricea de confuzie ────────────────────────────
        ax_mc = self.axe_test[0]
        mc = rezultate["matrice_confuzie"]  # [[TN, FP], [FN, TP]]

        im = ax_mc.imshow(mc, interpolation="nearest",
                          cmap=plt.cm.Blues, vmin=0)
        ax_mc.set_title("Matrice de confuzie", fontsize=9, fontweight="bold")
        ax_mc.set_xticks([0, 1])
        ax_mc.set_yticks([0, 1])
        ax_mc.set_xticklabels(["Pred. Benign", "Pred. Malign"], fontsize=7)
        ax_mc.set_yticklabels(["Real Benign", "Real Malign"], fontsize=7)
        ax_mc.set_xlabel("Predicție", fontsize=8)
        ax_mc.set_ylabel("Realitate", fontsize=8)

        # Valorile numerice în celule
        max_val = mc.max()
        for i in range(2):
            for j in range(2):
                culoare_text = "white" if mc[i, j] > max_val / 2 else "black"
                ax_mc.text(j, i, str(mc[i, j]),
                           ha="center", va="center",
                           color=culoare_text, fontsize=14, fontweight="bold")

        self.fig_test.colorbar(im, ax=ax_mc, fraction=0.046)

        # ── Grafic dreapta: Bar chart cu metrici ────────────────────────────
        ax_met = self.axe_test[1]
        etichete_met = ["Acuratețe", "Precizie", "Recall", "F1-Score"]
        valori_met   = [
            rezultate["acuratete"],
            rezultate["precizie"],
            rezultate["recall"],
            rezultate["f1"]
        ]
        culori_met = ["#1976D2", "#4CAF50", "#FF9800", "#9C27B0"]

        bare = ax_met.bar(etichete_met, valori_met, color=culori_met,
                          edgecolor="white", width=0.5)
        ax_met.set_ylim(0, 1.1)
        ax_met.set_title("Metrici de performanță", fontsize=9, fontweight="bold")
        ax_met.set_ylabel("Valoare")

        # Etichete procentuale pe bare
        for bara, val in zip(bare, valori_met):
            ax_met.text(bara.get_x() + bara.get_width() / 2,
                        bara.get_height() + 0.02,
                        f"{val*100:.1f}%",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax_met.tick_params(axis="x", labelsize=7)
        self.fig_test.tight_layout()
        self.canvas_test.draw()

    # =========================================================================
    # Logica: Salvare și Încărcare model
    # =========================================================================
    def _salveaza_model(self):
        """Deschide un dialog pentru salvarea modelului în format JSON."""
        if self.model is None:
            messagebox.showwarning("Model lipsă", "Nu există un model antrenat de salvat!")
            return

        cale = filedialog.asksaveasfilename(
            title="Salvează modelul",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Toate fișierele", "*.*")]
        )
        if not cale:
            return

        # Colectăm hiperparametrii curenți din GUI
        hiperparametri = {
            "straturi_ascunse": self.entry_straturi.get(),
            "activare":         self.combo_activare.get(),
            "max_epoci":        self.entry_epoci.get(),
            "rata_invatare":    self.entry_rata.get(),
            "eroare_admisa":    self.entry_eroare.get(),
            "init_ponderi":     self.combo_init.get(),
        }

        try:
            self.model.salveaza(cale, self.params_norm, hiperparametri)
            messagebox.showinfo("Salvat", f"Modelul a fost salvat în:\n{cale}")
        except Exception as e:
            messagebox.showerror("Eroare la salvare", str(e))

    def _incarca_model(self):
        """Deschide un dialog pentru încărcarea unui model din JSON."""
        cale = filedialog.askopenfilename(
            title="Încarcă modelul",
            filetypes=[("JSON", "*.json"), ("Toate fișierele", "*.*")]
        )
        if not cale:
            return

        try:
            model, params_norm, hiperparametri = MLP.incarca(cale)
            self.model = model
            self.params_norm = params_norm

            # Populăm câmpurile GUI cu hiperparametrii din model
            if hiperparametri:
                self.entry_straturi.delete(0, tk.END)
                self.entry_straturi.insert(0, hiperparametri.get("straturi_ascunse", "8"))

                act = hiperparametri.get("activare", "sigmoid")
                if act in ["sigmoid", "tanh"]:
                    self.combo_activare.set(act)

                self.entry_epoci.delete(0, tk.END)
                self.entry_epoci.insert(0, hiperparametri.get("max_epoci", "100"))

                self.entry_rata.delete(0, tk.END)
                self.entry_rata.insert(0, hiperparametri.get("rata_invatare", "0.1"))

                self.entry_eroare.delete(0, tk.END)
                self.entry_eroare.insert(0, hiperparametri.get("eroare_admisa", "0.001"))

                init_v = hiperparametri.get("init_ponderi", "Xavier")
                self.combo_init.set(init_v)

            self.lbl_stare.config(text="Stare: ✓ model încărcat")
            messagebox.showinfo("Încărcat", f"Modelul a fost încărcat din:\n{cale}")

            # Evaluăm automat modelul încărcat
            if self.date_test is not None:
                self._evalueaza_model()

        except Exception as e:
            messagebox.showerror("Eroare la încărcare", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Punctul de intrare al aplicației
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicatieMLP(root)
    root.mainloop()
