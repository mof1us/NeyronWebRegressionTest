import json
import torch
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from Web import Web


def load_emnist_data(batch_size=64):
    emnist = EMNIST("./data", split="digits", train=True, download=True,
                    transform=transforms.ToTensor())
    loader = DataLoader(emnist, batch_size=2048, shuffle=False,
                        num_workers=4, pin_memory=True)
    rows = []
    for imgs, targets in loader:
        # imgs:  (B, 1, 28, 28)  → делаем (B, 784)
        flat = imgs.view(imgs.size(0), -1)  # (B, 784)

        # targets: (B,) целые метки 0–9 → one-hot (B, 10)
        onehot = F.one_hot(targets, num_classes=10).float()

        # объединяем и добавляем в список
        rows.append(torch.cat([flat, onehot], dim=1))  # (B, 794)

    # 3. Собираем всё в единый двумерный тензор и переводим в NumPy
    data = torch.cat(rows, dim=0).cpu().numpy()  # shape = (N, 794)
    # 4. Сохраняем в CSV
    file_path = "edu_file.csv"
    open(file_path, 'w').close()  # Очищаем файл перед записью
    np.savetxt(file_path, data, delimiter=',', fmt='%.10f')
    print(data.shape)

if __name__ == "__main__":
    # load_emnist_data()
    config = json.load(open("config.json"))
    web = Web(config["first_layer_size"], config["count_of_hidden_layers"], config["last_layer_size"], config["batch_size"], config["weight_decay"])
    web.start_education()
    while True:
        data = list(map(int, input("Введите точку x\n").split()))
        res = web.predict(np.array(data))
        print(res)
