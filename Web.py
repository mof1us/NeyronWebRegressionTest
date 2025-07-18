import time

import numpy as np
import logging
from tqdm import tqdm

class Web:
    def __init__(self, first_layer_size: int, count_of_hidden_layers: int, last_layer_size: int, batch_size: int = 1, weight_decay: int=0.0001,  learning_data_size: int = 240000, learning_rate: int = 1e-4):
        self.dl_dw_l_arr = None
        self.disp = None
        self.curr_batch = None
        self.curr_sigma = None
        self.weight_decay = weight_decay
        self.first_layer_size = first_layer_size
        self.count_of_hidden_layers = count_of_hidden_layers
        self.last_layer_size = last_layer_size
        self.batch_size = batch_size
        self.learning_data_size = learning_data_size
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # self.logger.addHandler(console_handler)
        #sdgjksbhgjisgnhas
        self.layer_sizes: list[int] = [self.first_layer_size]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(1, count_of_hidden_layers + 1):
            r = (last_layer_size / first_layer_size) ** (1 / (count_of_hidden_layers + 1))
            layer_size = int(first_layer_size * (r ** (i + 1)))
            print(layer_size)
            self.layer_sizes.append(layer_size)
            self.biases.append(np.random.randn(layer_size, 1))
        self.biases.append(np.random.randn(last_layer_size, 1))
        self.layer_sizes.append(last_layer_size)
        self.saved_H = []
        self.saved_Z = []

        self.y_results = np.zeros((batch_size, self.last_layer_size), dtype=np.float64)

        self.education_matrix = []
        self.education_matrix_answers = []


    def load_learning_data(self):
        self.education_matrix = np.zeros((self.learning_data_size, self.first_layer_size, 1))
        self.education_matrix_answers = np.zeros((self.learning_data_size, self.last_layer_size), dtype=np.float64)
        loaded_data = open("edu_file.csv").readlines()
        for i in range(self.learning_data_size):
            example = loaded_data[i].strip().split(",")
            self.education_matrix[i] = np.array([float(x) for x in example[:self.first_layer_size]]).reshape(-1, 1)
            self.education_matrix_answers[i] = np.array([float(x) for x in example[self.first_layer_size:]])
        self.logger.info("Loaded learning data")

        ## Далее нормализация данных
        X = self.education_matrix.reshape(self.learning_data_size, -1)  # (N, d)
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-8
        self.education_matrix = ((X - self.x_mean) / self.x_std).reshape(
            self.learning_data_size, self.first_layer_size, 1)
        y = self.education_matrix_answers
        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-8
        self.education_matrix_answers = (y - self.y_mean) / self.y_std

    def start_education(self):
        self.load_learning_data()
        self.init_weights()

        N = self.education_matrix.shape[0]

        for epoch in range(1, 50):  # Количество эпох обучения
            idx = np.random.permutation(N)
            for i in tqdm(range(0, self.learning_data_size, self.batch_size), desc=f"epoch {epoch} Training progress"):
                if (self.batch_size + i >= self.learning_data_size ):
                    continue
                batch_idx = idx[i:i + self.batch_size]
                self.forward_pass(batch_idx)
                self.back_propagation(batch_idx)
                self.weight_recalculation()


    def init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.normal(0, np.sqrt(2 / self.layer_sizes[i]), size=(self.layer_sizes[i], self.layer_sizes[i + 1])))
        self.logger.info("Initialized weights")
        # Определяем все веса. Массив весов имеет следующий вид: self.weights[Номер слоя][Номер нейрона в слое][Номер нейрона в следующем слое]

    def activation_function(self, x):
        alpha = 0.01  # Коэффициент наклона для отрицательной части
        return np.where(x >= 0, x, alpha * x)  # Применяем ReLU поэлементно для матрицы

    def forward_pass(self, batch_idx):
        self.saved_H = []
        self.saved_Z = []

        H_l = self.education_matrix[batch_idx].reshape(len(batch_idx), self.first_layer_size)
        self.saved_H.append(H_l)
        for current_layer in range(0, self.count_of_hidden_layers):
            W_l = self.weights[current_layer]
            Z_l = np.dot(H_l, W_l)
            b_l_t = self.biases[current_layer].T
            B_1 = np.ones((len(Z_l), 1))
            Z_l = Z_l + np.dot(B_1, b_l_t)  # Добавляем смещение
            activated_layer = self.activation_function(Z_l)  # Активируем слой
            self.saved_Z.append(Z_l)
            H_l = activated_layer
            self.saved_H.append(H_l)
        W_l = self.weights[self.count_of_hidden_layers]
        Z_l = np.dot(H_l, W_l)
        b_l_t = self.biases[self.count_of_hidden_layers].T

        B_1 = np.ones((len(Z_l), 1))

        Z_l = Z_l + np.dot(B_1, b_l_t)  # Добавляем смещение
        self.saved_H.append(Z_l)
        self.y_results = Z_l
        self.logger.info("Finished forward pass")



    def loss_gradient_dl_dy(self, batch_idx):
        grad = 2 * (self.y_results - self.education_matrix_answers[batch_idx]) / (self.batch_size * self.last_layer_size)
        return grad

    def d_activation_d_x(self, z):
        alpha = 0.01
        return np.where(z >= 0, 1.0, alpha)

    def back_propagation(self, batch_idx):
        self.curr_sigma = self.loss_gradient_dl_dy(batch_idx)
        self.dl_dw_l_arr = []
        self.dl_db_l_arr = []
        for layer in range(self.count_of_hidden_layers, -1, -1):
            H_l_t = self.saved_H[layer].T
            dl_dw_l = np.dot(H_l_t, self.curr_sigma)  # Вычисляем градиент по весам
            dl_db_l = np.sum(self.curr_sigma, axis=0, keepdims=True)  # Вычисляем градиент по смещениям
            self.dl_dw_l_arr.append(dl_dw_l)
            self.dl_db_l_arr.append(dl_db_l)
            if layer > 0:
                self.curr_sigma = np.multiply(np.dot(self.curr_sigma, self.weights[layer].T), self.d_activation_d_x(self.saved_Z[layer - 1]))
        self.dl_dw_l_arr.reverse()
        self.dl_db_l_arr.reverse()
        self.logger.info("Finished back propagation")


    def weight_recalculation(self):
        for layer in range(self.count_of_hidden_layers, -1, -1):
            self.weights[layer] -= (self.dl_dw_l_arr[layer]) * self.learning_rate + self.weight_decay * self.weights[layer] * self.learning_rate

            self.biases[layer] -= self.dl_db_l_arr[layer].T * self.learning_rate
        self.logger.info("Finished weight recalculation")

    def predict(self, X: np.ndarray, *, auto_batch=True, load_if_needed=False):
        """
        X  : np.ndarray  shape (N, first_layer_size)  or (first_layer_size,)
        Returns ndarray shape (N, last_layer_size)
        """
        # — 0. если веса пусты — загружаем чек-пойнт
        if load_if_needed and not self.weights:
            self.load()

        # — 1. приведение формы
        if X.ndim == 1:
            X = X.reshape(1, -1)                # (d,) → (1,d)
        assert X.shape[1] == self.first_layer_size,\
            f"Expected {self.first_layer_size} features, got {X.shape[1]}"
        X = (X - self.x_mean) / self.x_std
        N = X.shape[0]
        results = []

        # — 2. батчевый проход
        batch = self.batch_size if auto_batch else N
        for i in range(0, N, batch):
            xb = X[i:i + batch]                 # (B,d)
            Z = xb                              # ← H0
            for l in range(self.count_of_hidden_layers + 1):
                W = self.weights[l]             # (n_{l-1}, n_l)
                Z = Z @ W                       # матр. умножение
                Z += self.biases[l].T       # broadcast bias
                if l < self.count_of_hidden_layers:
                    Z = self.activation_function(Z)  # ReLU/Leaky ReLU
                # последний слой оставляем линейным
            results.append(Z)                   # (B, last_layer_size)

        Y_pred = np.vstack(results)             # обратно в (N, k)
        Y_pred = Y_pred * self.y_std + self.y_mean
        # — 3. обратная денормализация (если делали на обучении)
        # Y_pred = self.denorm(Y_pred)

        return Y_pred.squeeze()                 # (N,) если k==1