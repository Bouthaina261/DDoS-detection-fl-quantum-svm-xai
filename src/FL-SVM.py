import os
import glob
import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# 0) CONFIG
# =========================================================
SEED = 42
NUM_CLIENTS = 6
NUM_ROUNDS = 10

LOCAL_EPOCHS = 3
BATCH_SIZE = 512

ALPHA = 1e-4
ETA0 = 1e-2
TEST_SIZE_GLOBAL = 0.2

rng = np.random.default_rng(SEED)

# =========================================================
# 1) LOAD (1 CSV ou dossier)
# =========================================================
FILE_OR_DIR = "/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

def load_cicddos(file_or_dir: str) -> pd.DataFrame:
    if os.path.isdir(file_or_dir):
        paths = sorted(glob.glob(os.path.join(file_or_dir, "*.csv")))
        if len(paths) == 0:
            raise FileNotFoundError(f"Aucun CSV trouvé dans: {file_or_dir}")
        dfs = [pd.read_csv(p) for p in paths]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(file_or_dir)

    df.columns = df.columns.str.strip()
    return df

df = load_cicddos(FILE_OR_DIR)

# =========================================================
# 2) CLEAN + LABEL (BENIGN=0, ATTACK=1)
# =========================================================
df = df.replace([np.inf, -np.inf], np.nan)

if "Label" not in df.columns:
    raise ValueError(
        f"Colonne 'Label' introuvable. Colonnes dispo: {df.columns.tolist()[:40]} ..."
    )

df["Label"] = df["Label"].astype(str).str.strip()
y = np.where(df["Label"].str.upper().eq("BENIGN"), 0, 1).astype(np.int64)

# =========================================================
# 2bis) FEATURES: 8 features
# =========================================================
features = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow IAT Mean", "Flow Bytes/s", "Flow Packets/s",
    "Fwd IAT Mean", "Bwd IAT Mean"
]

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(
        f"Features manquantes dans le CSV: {missing}\n"
        f"Colonnes disponibles (extrait): {df.columns.tolist()[:80]}"
    )

X = df[features].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).values.astype(np.float32)

# =========================================================
# 3) GLOBAL SPLIT + SCALING (sans fuite)
# =========================================================
X_train_all, X_test_global, y_train_all, y_test_global = train_test_split(
    X, y, test_size=TEST_SIZE_GLOBAL, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_all = scaler.fit_transform(X_train_all).astype(np.float32)
X_test_global = scaler.transform(X_test_global).astype(np.float32)

# =========================================================
# 4) PARTITION TRAIN entre clients
# (attaque divisée en 6) + (normal divisé en 6) puis concat ✅
# =========================================================
attack_idx = np.where(y_train_all == 1)[0]
normal_idx = np.where(y_train_all == 0)[0]

rng.shuffle(attack_idx)
rng.shuffle(normal_idx)

attack_splits = np.array_split(attack_idx, NUM_CLIENTS)
normal_splits = np.array_split(normal_idx, NUM_CLIENTS)

clients_idx = []
for i in range(NUM_CLIENTS):
    idx = np.concatenate([attack_splits[i], normal_splits[i]])
    rng.shuffle(idx)
    clients_idx.append(idx)

# =========================================================
# 5) SVM Model (Linear SVM via SGDClassifier)
# =========================================================
def init_svm():
    return SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=ALPHA,
        learning_rate="constant",
        eta0=ETA0,
        max_iter=1,
        tol=None,
        average=False,
        random_state=SEED,
    )

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def hinge_loss_binary(y_true01, margins):
    y_signed = (2 * y_true01 - 1).astype(np.float32)
    return float(np.mean(np.maximum(0.0, 1.0 - y_signed * margins)))

# =========================================================
# 6) Flower Client (SVM)
# -> Accuracy client calculée dans evaluate() (comme Quantum) ✅
# -> On renvoie aussi "cid" dans metrics ✅
# =========================================================
class SVMClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.rng = np.random.default_rng(SEED + cid)

        idx = clients_idx[cid]
        Xc = X_train_all[idx]
        yc = y_train_all[idx]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            Xc, yc, test_size=0.2, random_state=SEED, stratify=yc
        )

        self.model = init_svm()

        # init coef_/intercept_/classes_
        n_init = min(64, len(self.X_train))
        self.model.partial_fit(
            self.X_train[:n_init],
            self.y_train[:n_init],
            classes=np.array([0, 1]),
        )

    def get_parameters(self, config):
        return [self.model.coef_.copy(), self.model.intercept_.copy()]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0].copy()
        self.model.intercept_ = parameters[1].copy()
        self.model.classes_ = np.array([0, 1])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        batch_size = int(config.get("batch_size", BATCH_SIZE))

        n = len(self.X_train)
        for _ in range(local_epochs):
            perm = self.rng.permutation(n)
            Xs, ys = self.X_train[perm], self.y_train[perm]

            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                self.model.partial_fit(xb, yb)

        return self.get_parameters({}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # ✅ appelé à chaque round si fraction_evaluate=1.0
        self.set_parameters(parameters)

        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        margins = self.model.decision_function(self.X_test)
        loss = hinge_loss_binary(self.y_test, margins)

        # ✅ IMPORTANT: renvoyer cid (0..NUM_CLIENTS-1) pour affichage côté serveur
        return float(loss), len(self.X_test), {
            "cid": int(self.cid),
            "accuracy": float(acc),
            "f1": float(f1),
        }

def client_fn(cid: str):
    return SVMClient(int(cid)).to_client()

# =========================================================
# 7) Évaluation globale serveur sur test global
# =========================================================
server_model = init_svm()
n_init_server = min(64, len(X_train_all))
server_model.partial_fit(
    X_train_all[:n_init_server], y_train_all[:n_init_server], classes=np.array([0, 1])
)

def server_evaluate(server_round, parameters, config):
    server_model.coef_ = parameters[0].copy()
    server_model.intercept_ = parameters[1].copy()
    server_model.classes_ = np.array([0, 1])

    y_pred = server_model.predict(X_test_global)
    acc, prec, rec, f1 = compute_metrics(y_test_global, y_pred)

    margins = server_model.decision_function(X_test_global)
    loss = hinge_loss_binary(y_test_global, margins)

    return float(loss), {
        "global_accuracy": float(acc),
        "global_precision": float(prec),
        "global_recall": float(rec),
        "global_f1": float(f1),
    }

# =========================================================
# 8) STRATEGY: FedAvg custom -> imprime accuracy client à chaque round ✅
# -> affiche "Client 1..Client N" (au lieu des IDs Ray) ✅
# =========================================================
from flwr.server.strategy import FedAvg

class FedAvgPrintClientAcc(FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        if results:
            print(f"\n📊 ROUND {server_round} | Client accuracies:")

            # Trier par cid pour avoir un affichage stable Client 1..N
            rows = []
            for _, evaluate_res in results:
                cid = int(evaluate_res.metrics.get("cid", -1))
                acc = evaluate_res.metrics.get("accuracy", None)
                f1 = evaluate_res.metrics.get("f1", None)
                rows.append((cid, acc, f1))

            rows.sort(key=lambda t: t[0])

            for cid, acc, f1 in rows:
                client_name = f"Client {cid}" if cid >= 0 else "Client ?"
                if acc is None:
                    print(f"   - {client_name}: accuracy=N/A")
                else:
                    if f1 is None:
                        print(f"   - {client_name}: accuracy={acc:.4f}")
                    else:
                        print(f"   - {client_name}: accuracy={acc:.4f} | f1={f1:.4f}")

        return super().aggregate_evaluate(server_round, results, failures)

# =========================================================
# 9) Simulation
# =========================================================
print(f"🚀 Démarrage FL SVM (clients={NUM_CLIENTS}, rounds={NUM_ROUNDS}, local_epochs={LOCAL_EPOCHS}, batch={BATCH_SIZE})")
print("✅ Features utilisées:", features)
print(f"✅ SVM params: alpha={ALPHA}, eta0={ETA0}, lr=constant")
print("✅ Évaluation: GLOBAL (serveur) + accuracy clients à chaque round")

strategy = FedAvgPrintClientAcc(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,

    # ✅ IMPORTANT: pour appeler evaluate() chez tous les clients à chaque round
    fraction_evaluate=1.0,
    min_evaluate_clients=NUM_CLIENTS,

    # ✅ Évaluation centralisée serveur (metrics globales)
    evaluate_fn=server_evaluate,

    on_fit_config_fn=lambda rnd: {
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
    },

    # (optionnel) si tu veux passer server_round aux clients (pas nécessaire ici)
    on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
)

history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

print("\n✅ Terminé")
print("Centralized losses:", history.losses_centralized)
print("Centralized metrics:", history.metrics_centralized)

# =========================================================
# 10) EXTRACTION pour PLOTS (GLOBAL ONLY)
# =========================================================
loss_tuples = history.losses_centralized or []
rounds_loss = [r for (r, _) in loss_tuples]
global_losses = [v for (_, v) in loss_tuples]

acc_tuples = (history.metrics_centralized or {}).get("global_accuracy", [])
rounds_acc = [r for (r, _) in acc_tuples]
global_accuracies = [v for (_, v) in acc_tuples]

f1_tuples = (history.metrics_centralized or {}).get("global_f1", [])
rounds_f1 = [r for (r, _) in f1_tuples]
global_f1s = [v for (_, v) in f1_tuples]

# =========================================================
# 11) PLOTS
# =========================================================
plt.figure()
plt.plot(rounds_loss, global_losses)
plt.xlabel("Round")
plt.ylabel("Global Loss")
plt.title("Global Loss Evolution (Federated Learning + SVM)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(rounds_acc, global_accuracies)
plt.xlabel("Round")
plt.ylabel("Global Accuracy")
plt.title("Global Accuracy Evolution (Federated Learning + SVM)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(rounds_f1, global_f1s)
plt.xlabel("Round")
plt.ylabel("Global F1-score")
plt.title("Global F1 Evolution (Federated Learning + SVM)")
plt.grid(True)
plt.show()

# =========================================================
# 12) SHAP (sur modèle global final)
# =========================================================
final_model = server_model  # après le dernier round, server_evaluate a mis à jour server_model

X_test_df = pd.DataFrame(X_test_global, columns=features)

n_sample = 10000
X_explain = X_test_df.sample(n=n_sample, random_state=SEED) if len(X_test_df) > n_sample else X_test_df

X_train_df = pd.DataFrame(X_train_all, columns=features)
bg_size = 2000
background = X_train_df.sample(n=bg_size, random_state=SEED) if len(X_train_df) > bg_size else X_train_df

explainer = shap.LinearExplainer(final_model, background, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_explain)

plt.title("SHAP - Importance globale des features (SVM fédéré)")
shap.summary_plot(shap_values, X_explain, plot_type="bar", show=True)

plt.title("SHAP - Impact des features sur la décision (SVM fédéré)")
shap.summary_plot(shap_values, X_explain, show=True)

i = 0
shap_exp = shap.Explanation(
    values=shap_values[i],
    base_values=explainer.expected_value,
    data=X_explain.iloc[i].values,
    feature_names=features
)
plt.title("SHAP - Waterfall (1 flow expliqué)")
shap.plots.waterfall(shap_exp, show=True)

print("✅ SHAP terminé (SVM fédéré) - explications basées sur decision_function (marges).")