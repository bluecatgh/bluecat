
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class LogisticRegressionSGD(nn.Module):

    def __init__(self, input_dim, learning_rate=0.05,
                 epochs=200, batch_size=128):
        super().__init__()

        self.linear = nn.Linear(input_dim, 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []

    def forward(self, X):
        # Return linear scores (logits)
        return self.linear(X)

    def fit(self, X, y):

        torch.manual_seed(42)

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(
            y, dtype=torch.float32
        ).reshape(-1, 1)

        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        loss_fn = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate
        )

        self.loss_history = []

        for epoch in range(self.epochs):

            self.train()
            total_loss = 0.0

            for X_batch, y_batch in loader:

                logits = self.forward(X_batch)
                loss = loss_fn(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(X_batch)

            avg_loss = total_loss / len(dataset)
            self.loss_history.append(avg_loss)

            if (epoch + 1) % 25 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        return self

    def predict_proba(self, X):

        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32)

        with torch.no_grad():
            probabilities = torch.sigmoid(
                self.forward(X)
            )

        return probabilities.numpy().flatten()

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)

