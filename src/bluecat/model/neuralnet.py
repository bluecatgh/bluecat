"""TorchNet for CPE586 Homework 5, Question 11.1."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


class TorchNet(nn.Module):
    """Diabetes classifier with hidden layers of 4 and 3 neurons."""

    def __init__(self, input_dim=8, output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        self.loss_history = []
        self.network = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, output_dim),
        )

    def forward(self, x):
        return self.network(x)

    def standardize_data(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train, epochs=1000, lr=0.01):
        features = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        targets = torch.tensor(np.asarray(y_train), dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_history = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            loss = criterion(self(features), targets)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if (epoch + 1) % 200 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Neural Network Training Loss")
        plt.grid(True)
        plt.show()

    def save_onnx(self, model_path, scaler_path):
        self.eval()
        dummy_input = torch.randn(1, self.input_dim)
        torch.onnx.export(
            self,
            (dummy_input,),
            model_path,
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
        )
        joblib.dump(self.scaler, scaler_path)
        print("ONNX model and scaler saved successfully.")

    @staticmethod
    def predict_saved(model_path, scaler_path, new_data):
        import onnxruntime as ort

        loaded_scaler = joblib.load(scaler_path)
        X_new = loaded_scaler.transform(new_data)
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        logits = session.run(
            None, {input_name: X_new.astype(np.float32)}
        )[0]
        return np.argmax(logits, axis=1)
