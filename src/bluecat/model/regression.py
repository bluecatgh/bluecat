import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class LinearRegression(nn.Module):

    def __init__(self, learning_rate=0.01, epochs=5000):
        super().__init__()

        # Learning rate and number of epochs
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize model parameters
        self.w0 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        self.w1 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        # SGD optimizer
        self.optimizer = torch.optim.SGD(
            [self.w0, self.w1],
            lr=self.learning_rate
        )

        # Mean Squared Error loss
        self.loss_fn = nn.MSELoss()

        # Store training history
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

        # Store R^2 after fitting
        self.r2_ = None


    def forward(self, x):
        # Linear model: y = w0 + w1*x
        return self.w0 + self.w1 * x


    def fit(self, x_train, y_train, x_test, y_test):

        # Convert data into PyTorch tensors
        x_train = torch.as_tensor(
            x_train,
            dtype=torch.float32
        ).reshape(-1)

        y_train = torch.as_tensor(
            y_train,
            dtype=torch.float32
        ).reshape(-1)

        x_test = torch.as_tensor(
            x_test,
            dtype=torch.float32
        ).reshape(-1)

        y_test = torch.as_tensor(
            y_test,
            dtype=torch.float32
        ).reshape(-1)

        # Clear previous history
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

        # Training using gradient descent
        for epoch in range(self.epochs):

            # Forward pass
            y_pred = self.forward(x_train)

            # Calculate MSE loss
            loss = self.loss_fn(y_pred, y_train)

            # Clear previous gradients
            self.optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update w0 and w1
            self.optimizer.step()

            # Store intermediate values
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        # Compute R^2 using test data
        with torch.no_grad():

            y_test_pred = self.forward(x_test)

            ss_res = torch.sum(
                (y_test - y_test_pred) ** 2
            )

            ss_tot = torch.sum(
                (y_test - torch.mean(y_test)) ** 2
            )

            self.r2_ = (1 - ss_res / ss_tot).item()

        print(f"w0 = {self.w0.item():.6f}")
        print(f"w1 = {self.w1.item():.6f}")
        print(f"Test R^2 = {self.r2_:.6f}")

        return self


    def predict(self, x):

        x = torch.as_tensor(
            x,
            dtype=torch.float32
        ).reshape(-1)

        with torch.no_grad():
            y_pred = self.forward(x)

        return y_pred.numpy()


    def analysis_plot(self, x, y):

        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)

        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32
        )

        with torch.no_grad():
            y_pred = self.forward(x_tensor).numpy()

        # Sort x so regression line is drawn correctly
        order = np.argsort(x)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8)
        )

        # Original data and fitted regression line
        axes[0, 0].scatter(x, y)
        axes[0, 0].plot(
            x[order],
            y_pred[order]
        )
        axes[0, 0].set_xlabel("BCR")
        axes[0, 0].set_ylabel("Annual Production")
        axes[0, 0].set_title(
            "BCR vs Annual Production"
        )

        # w0 during training
        axes[0, 1].plot(self.w0_history)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("w0")
        axes[0, 1].set_title(
            "w0 During Training"
        )

        # w1 during training
        axes[1, 0].plot(self.w1_history)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("w1")
        axes[1, 0].set_title(
            "w1 During Training"
        )

        # Loss during training
        axes[1, 1].plot(self.loss_history)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MSE Loss")
        axes[1, 1].set_title(
            "Loss During Training"
        )

        plt.tight_layout()
        plt.show()


class CauchyRegression(nn.Module):

    def __init__(self, input_dim, c=1.0):
        super().__init__()

        self.linear = nn.Linear(
            input_dim, 1, dtype=torch.float64
        )
        self.c = c
        self.loss_history = []

    def forward(self, X):
        return self.linear(X)

    def cauchy_loss(self, y_pred, y_true):
        residual = (y_true - y_pred) / self.c

        loss = (self.c**2 / 2) * torch.log1p(
            residual**2
        )
        return torch.mean(loss)

    def fit(self, X, y, learning_rate=1.0, epochs=3000):

        X = torch.tensor(X, dtype=torch.float64)
        y = torch.tensor(
            y.reshape(-1, 1), dtype=torch.float64
        )

        # Standardize input features
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0, unbiased=False)
        X_scaled = (X - self.X_mean) / self.X_std

        # Initialize parameters
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(y.mean().item())

        # Optimizer
        optimizer = torch.optim.SGD(
            self.parameters(), lr=learning_rate
        )

        self.loss_history = []

        # Training loop
        for epoch in range(epochs):

            y_pred = self.forward(X_scaled)
            loss = self.cauchy_loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())

            if (epoch + 1) % 500 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Loss: {loss.item():.6f}"
                )

        with torch.no_grad():
            self.final_loss = self.cauchy_loss(
                self.forward(X_scaled), y
            ).item()

        return self

    def predict(self, X):

        X = torch.tensor(X, dtype=torch.float64)
        X_scaled = (X - self.X_mean) / self.X_std

        with torch.no_grad():
            return self.forward(X_scaled).numpy().flatten()

    def get_coefficients(self):

        with torch.no_grad():
            w = self.linear.weight.flatten() / self.X_std
            w0 = self.linear.bias.item() - torch.dot(
                w, self.X_mean
            ).item()

        return w0, w.numpy()
