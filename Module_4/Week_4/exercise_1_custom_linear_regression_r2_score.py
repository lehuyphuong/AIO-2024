import numpy as np


def get_columns(data, index):
    result = [row[index] for row in data]
    return result


def load_data_from_file(filename="SalesPrediction.csv"):
    data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1)

    tv_data = get_columns(data, 0)
    radio_data = get_columns(data, 1)
    newspaper_data = get_columns(data, 2)
    sale_data = get_columns(data, 3)

    features_x = np.array([[1, x1, x2, x3]
                           for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)])
    sales_y = np.array(sale_data)

    return features_x, sales_y


class CustomLinearRegression:
    def __init__(self, x_data, y_target, learning_rate=0.01, num_epochs=10000):
        self.num_samples = x_data.shape[0]
        self.x_data = np.c_[np.ones((self.num_samples, 1)), x_data]
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epoches = num_epochs

        # Initial weights
        self.theta = np.random.randn(self.x_data.shape[1], 1)
        self.losses = []

    def compute_loss(self, y_pred, y_target):
        loss = np.multiply((y_target - y_pred), (y_target-y_pred))
        return loss

    def predict(self, x_data):
        y_pred = x_data.dot(self.theta)
        return y_pred

    def fit(self):
        for epoch in range(self.num_epoches):

            y_pred = self.predict(self.x_data)

            loss = self.compute_loss(y_pred=y_pred, y_target=self.y_target)
            self.losses.append(loss)

            loss_grd = 2*(y_pred-self.y_target)/self.num_samples
            gradients = self.x_data.T.dot(loss_grd)

            self.theta = self.theta - self.learning_rate*gradients

            if (epoch % 50) == 0:
                print(f'Epoch: {epoch} - Loss: {loss}')

        return {
            'loss': sum(self.losses)/len(self.losses),
            'weight': self.theta
        }

    def r2score(self, y_pred, y):
        rss = np.sum((y_pred-y)**2)
        tss = np.sum((y-y.mean())**2)
        r2 = 1 - (rss/tss)
        return r2


if __name__ == "__main__":
    x_data, y_target = load_data_from_file(filename="SalesPrediction.csv")
    model = CustomLinearRegression(x_data, y_target)

    y_pred = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])

    r2_score = model.r2score(y_pred, y)
    print(r2_score)

    y_pred = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 5, 2, 4])

    r2_score = model.r2score(y_pred, y)
    print(r2_score)

