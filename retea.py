"""
retea.py - Implementarea de la zero a rețelei neuronale MLP (Perceptron Multistrat)
folosind exclusiv biblioteca NumPy (fără sklearn, fără pytorch/tensorflow).
Nu conține nicio dependență de interfața grafică.
"""

import json
import numpy as np


class MLP:
    """
    Clasa Perceptron Multistrat (Multilayer Perceptron).
    
    Arhitectură:
    - Strat de intrare: fix 30 neuroni (cele 30 de caracteristici ale tumorii)
    - Straturi ascunse: configurabile (de exemplu [8] sau [8, 4])
    - Strat de ieșire: 1 neuron (probabilitatea ca tumora să fie malignă, prag 0.5)
    """

    def __init__(
        self,
        dimensiuni_straturi,
        activare_ascunsa="sigmoid",
        metoda_init="xavier",
        seed=42,
    ):
        """
        Inițializează rețeaua neuronală.
        
        Parametri:
            dimensiuni_straturi: listă cu numărul de neuroni pe fiecare strat,
                                de ex: [30, 8, 1] sau [30, 8, 4, 1]
            activare_ascunsa: 'sigmoid' sau 'tanh'
            metoda_init: 'xavier' sau 'aleatoare' (aleatoare mici)
            seed: pentru reproductibilitatea inițializării ponderilor
        """
        self.dimensiuni_straturi = list(dimensiuni_straturi)
        self.activare_ascunsa = activare_ascunsa.lower()
        self.metoda_init = metoda_init.lower()
        self.seed = seed

        # Ponderile și bias-urile pentru fiecare conexiune dintre straturi
        self.ponderi = []
        self.biasuri = []

        # Istoricul erorilor MSE pe epoci pentru antrenare și validare
        self.istoric_erori_tren = []
        self.istoric_erori_val = []

        # Inițializăm ponderile și bias-urile
        self._initializeaza_parametri()

    def _initializeaza_parametri(self):
        """
        Inițializează ponderile și bias-urile pentru fiecare strat:
        - Xavier: deviație = sqrt(2 / (n_in + n_out))
        - Aleatoare mici: valori distribuite normal scalate cu 0.05
        - Bias-urile sunt inițializate cu 0
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.ponderi = []
        self.biasuri = []

        # Parcurgem fiecare pereche de straturi consecutive
        num_straturi = len(self.dimensiuni_straturi)
        for i in range(num_straturi - 1):
            n_in = self.dimensiuni_straturi[i]
            n_out = self.dimensiuni_straturi[i + 1]

            if self.metoda_init == "xavier":
                # Inițializare Xavier / Glorot
                deviatie = np.sqrt(2.0 / (n_in + n_out))
                W = np.random.randn(n_in, n_out) * deviatie
            else:
                # Inițializare aleatoare cu valori mici
                W = np.random.randn(n_in, n_out) * 0.05

            # Bias-urile încep de la 0
            b = np.zeros((1, n_out))

            self.ponderi.append(W)
            self.biasuri.append(b)

    # -------------------------------------------------------------
    # Funcții de activare și derivatele lor
    # -------------------------------------------------------------
    def _sigmoid(self, net, a=1.0, theta=0.0):
        """
        Funcția sigmoidă conform cerinței:
        f(net) = 1 / (1 + e^(-a * (net - theta))), cu a=1 și theta=0.
        Limităm valorile pentru a preveni overflow-ul numeric.
        """
        z = a * (net - theta)
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _derivata_sigmoid(self, valoare_activare):
        """
        Derivata funcției sigmoidă exprimată în funcție de ieșirea neuronului:
        f'(net) = f(net) * (1 - f(net))
        """
        return valoare_activare * (1.0 - valoare_activare)

    def _tanh(self, net):
        """
        Funcția tangentă hiperbolică:
        tanh(net) = (e^net - e^-net) / (e^net + e^-net)
        """
        return np.tanh(net)

    def _derivata_tanh(self, valoare_activare):
        """
        Derivata funcției tanh exprimată în funcție de ieșire:
        f'(net) = 1 - tanh(net)^2
        """
        return 1.0 - (valoare_activare ** 2)

    def _aplica_activare(self, net, tip):
        """Aplică funcția de activare specificată ('sigmoid' sau 'tanh')."""
        if tip == "sigmoid":
            return self._sigmoid(net)
        elif tip == "tanh":
            return self._tanh(net)
        else:
            raise ValueError(f"Funcție de activare necunoscută: {tip}")

    def _aplica_derivata(self, valoare_activare, tip):
        """Calculează derivata funcției de activare."""
        if tip == "sigmoid":
            return self._derivata_sigmoid(valoare_activare)
        elif tip == "tanh":
            return self._derivata_tanh(valoare_activare)
        else:
            raise ValueError(f"Funcție de activare necunoscută: {tip}")

    # -------------------------------------------------------------
    # Propagare înainte (Forward Propagation)
    # -------------------------------------------------------------
    def forward_exemplu(self, x):
        """
        Calculează ieșirea rețelei pentru un singur exemplu x.
        Neuronul calculează: net = suma_ponderata + bias
        Ieșire = activare(net).
        
        Pentru straturile ascunse se folosește funcția configurată (sigmoid/tanh).
        Pentru stratul final de ieșire se folosește mereu sigmoid (ieșire între 0 și 1).
        
        Returnează:
            activari: listă cu activările fiecărui strat [a_0, a_1, ..., a_L]
                      unde a_0 este intrarea x, iar a_L este predicția finală.
        """
        # Ne asigurăm că x are forma (1, nr_trăsături)
        a_curent = np.array(x, dtype=np.float64).reshape(1, -1)
        activari = [a_curent]

        num_straturi = len(self.ponderi)
        for l in range(num_straturi):
            W = self.ponderi[l]
            b = self.biasuri[l]

            # net = x * W + b (suma ponderată a intrărilor + bias)
            net = np.dot(a_curent, W) + b

            # Dacă este strat ascuns, folosim activarea ascunsă
            # Dacă este ultimul strat (ieșire), folosim mereu sigmoid
            if l < num_straturi - 1:
                a_curent = self._aplica_activare(net, self.activare_ascunsa)
            else:
                a_curent = self._aplica_activare(net, "sigmoid")

            activari.append(a_curent)

        return activari

    # -------------------------------------------------------------
    # Propagare înapoi și actualizare (Backpropagation + SGD)
    # -------------------------------------------------------------
    def pas_antrenare_exemplu(self, x, y, rata_invatare):
        """
        Execută un pas complet de antrenare pentru un singur exemplu:
        1. Forward propagation
        2. Calcul eroare MSE per pas: (y - y_pred)^2
        3. Backpropagation (calculul gradienților cu regula lanțului)
        4. Actualizarea ponderilor și a bias-urilor cu Gradient Descent
        
        Returnează:
            eroare_pas: eroarea pătratică pentru exemplul curent
        """
        # 1. Forward propagation
        activari = self.forward_exemplu(x)
        y_pred = activari[-1]

        # 2. Eroare per pas (MSE pe un singur exemplu)
        y_real = float(y)
        eroare_pas = float((y_real - y_pred[0, 0]) ** 2)

        # 3. Backpropagation
        num_straturi = len(self.ponderi)
        deltas = [None] * num_straturi

        # --- Stratul de ieșire (L-1) ---
        # Derivata MSE: dE/dy_pred = (y_pred - y_real)
        # Regula delta: delta_iesire = (y_pred - y_real) * f'(net_iesire)
        a_iesire = activari[-1]
        deriv_iesire = self._derivata_sigmoid(a_iesire)
        delta_iesire = (a_iesire - y_real) * deriv_iesire
        deltas[-1] = delta_iesire

        # --- Straturile ascunse (de la penultimul spre primul) ---
        for l in range(num_straturi - 2, -1, -1):
            W_urmator = self.ponderi[l + 1]
            delta_urmator = deltas[l + 1]
            a_curent = activari[l + 1]

            deriv_curent = self._aplica_derivata(a_curent, self.activare_ascunsa)

            # Propagăm eroarea înapoi prin ponderi și înmulțim cu derivata activării
            delta_curent = np.dot(delta_urmator, W_urmator.T) * deriv_curent
            deltas[l] = delta_curent

        # 4. Actualizare ponderi și bias-uri (Gradient Descent)
        for l in range(num_straturi):
            a_intrare = activari[l]
            delta = deltas[l]

            # Gradientul ponderilor: produs extern între intrarea stratului și delta
            grad_W = np.dot(a_intrare.T, delta)
            grad_b = delta

            # W_nou = W_vechi - rata * grad_W
            self.ponderi[l] -= rata_invatare * grad_W
            self.biasuri[l] -= rata_invatare * grad_b

        return eroare_pas

    # -------------------------------------------------------------
    # Predicție și Evaluare
    # -------------------------------------------------------------
    def prezice_probabilitati(self, X):
        """
        Calculează probabilitățile de ieșire pentru toate exemplele din X.
        Returnează un array 1D cu probabilitățile de a fi malign (1).
        """
        N = len(X)
        probabilitati = np.zeros(N)
        for i in range(N):
            activari = self.forward_exemplu(X[i])
            probabilitati[i] = activari[-1][0, 0]
        return probabilitati

    def prezice_clase(self, X, prag=0.5):
        """
        Prezice clasele (0 sau 1) aplicând pragul de decizie (implicit 0.5).
        Dacă probabilitatea >= 0.5 -> Malign (1), altfel -> Benign (0).
        """
        probabilitati = self.prezice_probabilitati(X)
        return (probabilitati >= prag).astype(int)

    def calculeaza_mse(self, X, y):
        """
        Calculează eroarea medie pătratică (MSE) pe un set întreg de date:
        MSE = (1 / N) * suma((y_real - y_pred)^2)
        """
        probabilitati = self.prezice_probabilitati(X)
        y_plat = np.array(y).flatten()
        mse = np.mean((y_plat - probabilitati) ** 2)
        return float(mse)

    # -------------------------------------------------------------
    # Bucla completă de antrenare pe epoci
    # -------------------------------------------------------------
    def antreneaza(
        self,
        X_tren,
        y_tren,
        X_val,
        y_val,
        max_epoci=100,
        rata_invatare=0.1,
        eroare_admisa=0.001,
        callback_epoca=None,
        oprire_solicitata=None,
    ):
        """
        Antrenează rețeaua neuronală pe epoci.
        La fiecare epocă:
        - Datele de antrenare sunt amestecate aleator
        - Se parcurge fiecare exemplu și se actualizează ponderile (SGD)
        - Se calculează eroarea MSE pe antrenare și pe validare
        - Se apelează callback_epoca pentru actualizarea graficului în timp real
        - Dacă eroarea scade sub eroare_admisa, antrenarea se oprește mai devreme
        - Dacă utilizatorul a apăsat butonul de oprire, antrenarea se întrerupe
        """
        self.istoric_erori_tren = []
        self.istoric_erori_val = []

        n_exemple = len(X_tren)

        for epoca in range(1, max_epoci + 1):
            # Verificăm dacă utilizatorul a cerut oprirea din GUI
            if oprire_solicitata and oprire_solicitata():
                break

            # 1. Amestecăm exemplele la fiecare epocă (stochastic shuffling)
            indici = np.random.permutation(n_exemple)

            suma_erori_pas = 0.0

            # 2. Actualizare după fiecare exemplu (SGD)
            for idx in indici:
                x_i = X_tren[idx]
                y_i = y_tren[idx]
                eroare_pas = self.pas_antrenare_exemplu(x_i, y_i, rata_invatare)
                suma_erori_pas += eroare_pas

            # 3. Calcul eroare MSE pe antrenare și validare
            mse_tren = suma_erori_pas / n_exemple
            mse_val = self.calculeaza_mse(X_val, y_val)

            self.istoric_erori_tren.append(mse_tren)
            self.istoric_erori_val.append(mse_val)

            # 4. Apelăm callback-ul pentru GUI (actualizare grafic în timp real)
            if callback_epoca:
                callback_epoca(epoca, mse_tren, mse_val)

            # 5. Criteriu de oprire timpurie dacă eroarea a scăzut sub pragul admis
            if mse_tren <= eroare_admisa:
                break

        return self.istoric_erori_tren, self.istoric_erori_val

    # -------------------------------------------------------------
    # Metrici de evaluare pe setul de test
    # -------------------------------------------------------------
    def evalueaza(self, X_test, y_test, prag=0.5):
        """
        Evaluează performanța rețelei pe setul de test:
        - Acuratețe = (TP + TN) / Total
        - Precizie  = TP / (TP + FP)
        - Recall    = TP / (TP + FN)
        - F1-Score  = 2 * (Precizie * Recall) / (Precizie + Recall)
        - Matricea de confuzie: [[TN, FP], [FN, TP]]
        """
        y_real = np.array(y_test).flatten().astype(int)
        y_pred = self.prezice_clase(X_test, prag=prag).astype(int)

        tp = int(np.sum((y_real == 1) & (y_pred == 1)))
        tn = int(np.sum((y_real == 0) & (y_pred == 0)))
        fp = int(np.sum((y_real == 0) & (y_pred == 1)))
        fn = int(np.sum((y_real == 1) & (y_pred == 0)))

        total = len(y_real)
        acuratete = (tp + tn) / total if total > 0 else 0.0
        precizie = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * (precizie * recall) / (precizie + recall)
            if (precizie + recall) > 0
            else 0.0
        )

        matrice_confuzie = np.array([[tn, fp], [fn, tp]])

        return {
            "acuratete": acuratete,
            "precizie": precizie,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "matrice_confuzie": matrice_confuzie,
        }

    # -------------------------------------------------------------
    # Salvare și Încărcare Model (JSON)
    # -------------------------------------------------------------
    def salveaza(self, cale_fisier, parametri_normalizare=None, hiperparametri=None):
        """
        Salvează modelul într-un fișier JSON conținând:
        - Arhitectura rețelei (dimensiuni_straturi, activare_ascunsa, metoda_init)
        - Ponderile și bias-urile antrenate (convertite în liste)
        - Hiperparametrii de antrenare
        - Parametrii de normalizare ai datelor (minim/maxim sau medie/deviație)
        """
        date_salvare = {
            "dimensiuni_straturi": self.dimensiuni_straturi,
            "activare_ascunsa": self.activare_ascunsa,
            "metoda_init": self.metoda_init,
            "ponderi": [w.tolist() for w in self.ponderi],
            "biasuri": [b.tolist() for b in self.biasuri],
            "hiperparametri": hiperparametri or {},
            "parametri_normalizare": parametri_normalizare or {},
        }

        with open(cale_fisier, "w", encoding="utf-8") as f:
            json.dump(date_salvare, f, indent=4)

    @classmethod
    def incarca(cls, cale_fisier):
        """
        Încarcă un model salvat din JSON și reconstruiește instanța MLP.
        Returnează:
            model_mlp: instanța rețelei cu ponderile restaurate
            parametri_normalizare: parametrii de scalare folosiți la antrenare
            hiperparametri: dicționarul cu hiperparametri
        """
        with open(cale_fisier, "r", encoding="utf-8") as f:
            date_incarcate = json.load(f)

        model = cls(
            dimensiuni_straturi=date_incarcate["dimensiuni_straturi"],
            activare_ascunsa=date_incarcate.get("activare_ascunsa", "sigmoid"),
            metoda_init=date_incarcate.get("metoda_init", "xavier"),
        )

        model.ponderi = [np.array(w, dtype=np.float64) for w in date_incarcate["ponderi"]]
        model.biasuri = [np.array(b, dtype=np.float64) for b in date_incarcate["biasuri"]]

        parametri_normalizare = date_incarcate.get("parametri_normalizare", {})
        hiperparametri = date_incarcate.get("hiperparametri", {})

        return model, parametri_normalizare, hiperparametri
