import random
import pandas as pd
from tqdm import tqdm
# Генерируем 100 000 случайных точек
num_tests = 100_000
coordinates = [(random.random() * 9, random.random() * 9) for _ in range(num_tests)]

test_data = [
    (x, y, x - 1, y, # x_left)
     x + 1, y)  # y_left
    for x, y in tqdm(coordinates)
]

df = pd.DataFrame(
    test_data,
)

# Сохраняем в CSV
file_path = "edu_file.csv"
open(file_path, 'w').close()  # Очищаем файл перед записью
df.to_csv(file_path, index=False, header=False)


