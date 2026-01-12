# =========================================================
# Quantum Federated Learning (Flower + PennyLane)
# =========================================================

import flwr as fl
import logging
logging.getLogger("flwr").setLevel(logging.INFO)

import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# =========================================================
# 1️⃣ Chargement et préparation des données
# =========================================================

file_path = "/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Binaire : attaque / normal
df["Label"] = df["Label"].apply(
    lambda x: 1 if "DoS" in x or "DDoS" in x else 0
)

features = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow IAT Mean", "Flow Bytes/s", "Flow Packets/s",
    "Fwd IAT Mean", "Bwd IAT Mean"
]

X = df[features].values
y = df["Label"].values

# Normalisation entre 0 et π
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)

# =========================================================
# 2️⃣ Partition ÉQUILIBRÉE pour 6 clients FL
# =========================================================

num_clients = 6

attack_idx = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]

np.random.shuffle(attack_idx)
np.random.shuffle(normal_idx)

attack_splits = np.array_split(attack_idx, num_clients)
normal_splits = np.array_split(normal_idx, num_clients)

clients_idx = []
for i in range(num_clients):
    idx = np.concatenate([attack_splits[i], normal_splits[i]])
    np.random.shuffle(idx)
    clients_idx.append(idx)

# =========================================================
# 3️⃣ Modèle quantique
# =========================================================

n_qubits = len(features)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    for i in range(n_qubits):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)

    return qml.expval(qml.PauliZ(0))

# =========================================================
# 4️⃣ Client Federated Learning quantique
# =========================================================

class QuantumClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        idx = clients_idx[cid]

        X_train, X_test, y_train, y_test = train_test_split(
            X[idx], y[idx], test_size=0.2, random_state=42
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.weights = pnp.random.random(
            (n_qubits, 3), requires_grad=True
        )

    def get_parameters(self, config):
        return [np.array(self.weights)]

    def fit(self, parameters, config):
        self.weights = pnp.array(parameters[0], requires_grad=True)
        lr = 0.1

        for x, target in zip(self.X_train[:30], self.y_train[:30]):
            def cost_fn(weights):
                pred = (quantum_circuit(x, weights) + 1) / 2
                return (pred - target) ** 2

            grad = qml.grad(cost_fn)(self.weights)
            self.weights = self.weights - lr * grad

        return [np.array(self.weights)], len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.weights = pnp.array(parameters[0], requires_grad=True)

        preds = []
        for x in self.X_test:
            pred = (quantum_circuit(x, self.weights) + 1) / 2
            preds.append(1 if pred > 0.5 else 0)

        acc = accuracy_score(self.y_test, preds)
        round_id = config.get("server_round", "N/A")

        print(
            f"📊 ROUND {round_id} | Client {self.cid} | Accuracy = {acc:.4f}"
        )

        # loss = 1 - accuracy
        return float(1 - acc), len(self.X_test), {"accuracy": acc}

# =========================================================
# 5️⃣ Création des clients
# =========================================================

def client_fn(cid: str):
    return QuantumClient(int(cid)).to_client()

# =========================================================
# 6️⃣ Lancement de la simulation FL (AVEC HISTORIQUE)
# =========================================================

print("🚀 Démarrage de la simulation Quantum Federated Learning")

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=6,
    min_evaluate_clients=6,
    min_available_clients=6,
)

server_config = fl.server.ServerConfig(num_rounds=25)

history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=6,
    config=server_config,
    strategy=strategy,
    client_resources={"num_cpus": 1},
)

print("✅ Simulation Quantum Federated Learning terminée")

# =========================================================
# 7️⃣ Global Accuracy & Loss (FIN DE SIMULATION)
# =========================================================

global_losses = [loss for _, loss in history.losses_distributed]
global_accuracies = [1 - loss for loss in global_losses]

rounds = range(1, len(global_losses) + 1)

# ---- Plot LOSS
plt.figure()
plt.plot(rounds, global_losses)
plt.xlabel("Round")
plt.ylabel("Global Loss")
plt.title("Global Loss Evolution (Federated Learning)")
plt.show()

# ---- Plot ACCURACY
plt.figure()
plt.plot(rounds, global_accuracies)
plt.xlabel("Round")
plt.ylabel("Global Accuracy")
plt.title("Global Accuracy Evolution (Federated Learning)")
plt.show()

# ---- Print summary
print("\n📌 Global Accuracy per round:")
for r, acc in zip(rounds, global_accuracies):
    print(f"🌍 Round {r}: Accuracy = {acc:.4f}")

print(f"\n🎯 Final Global Accuracy: {global_accuracies[-1]:.4f}")
