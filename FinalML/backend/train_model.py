import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model import StudentNet
from preprocess import fit_and_save_preprocessor, transform_df
import joblib

# Пути к файлам
csv_path = r"C:\Users\user\Desktop\FinalML\data\Student_Performance.csv"
prep_path = "preprocessor.joblib"
model_path = "student_net.pt"

# Словарь для кодирования оценок
GRADE_TO_IDX = {g:i for i,g in enumerate(sorted(['a','b','c','d','e','f']))}

if __name__ == '__main__':
    # Загружаем данные
    df = pd.read_csv(csv_path)
    
    # Подготавливаем препроцессор и сохраняем
    pre = fit_and_save_preprocessor(df, prep_path)
    
    # Преобразуем признаки в массив
    X = transform_df(df, pre)
    
    # Целевая переменная
    y = df['final_grade'].map(GRADE_TO_IDX).values

    # Приводим типы к float32 / int64
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Создаём датасет и DataLoader
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(dataset, batch_size=16, shuffle=True)

    # Создаём модель
    input_dim = X.shape[1]
    model = StudentNet(input_dim=input_dim, hidden_dim=64, n_classes=6)

    # Настраиваем устройство (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Обучение
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(dl.dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg:.4f}")

    # Сохраняем модель
    torch.save({'model_state_dict': model.state_dict(), 'input_dim': input_dim}, model_path)
    print("Saved model ->", model_path)