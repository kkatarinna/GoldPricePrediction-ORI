import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn import model_selection

test_num = 14

# Učitavanje podataka iz CSV fajla
df = pd.read_csv('data/Gold.csv')
df = df.iloc[::-1]

# Obrada N/A vrednosti
df.ffill(inplace=True)
df.bfill(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

timeseries = df[["Close/Last", "Volume", "Open", "High", "Low"]].values.astype('float32')
plt.plot(df.index, timeseries[:, 0])
plt.xlabel('Date')
plt.ylabel('Close/Last')
plt.title('Gold Prices Over Time')
plt.show()

mms = preprocessing.MinMaxScaler()
features = ['Volume', 'Open', 'High', 'Low']
X = df[features]
y = df[['Close/Last']].values

training_data_len = math.ceil(len(y) * 0.85)

training_scaled_data = mms.fit_transform(y)
train_data = training_scaled_data[0:training_data_len, :]

lookback = 1095  # ~ last 3 years data
X_train = []
y_train = []

for i in range(lookback, len(train_data)):
    X_train.append(train_data[i - lookback:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.linear(output[:, -1, :])  # Koristi samo izlaz iz poslednjeg vremenskog koraka
        return output

test_data = training_scaled_data[training_data_len - lookback:, :]

X_test = []
y_test = y[training_data_len:, :]

for i in range(lookback, len(test_data)):
    X_test.append(test_data[i - lookback:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Konvertovanje numpy array u torch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Dodajemo dimenziju
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # Dodajemo dimenziju

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Definišemo hiperparametre
input_size = X_train.shape[2]
hidden_size = 64
output_size = 1  # Predviđanje jedne tačke
eval_after = 10

model = LSTM(input_size, hidden_size, output_size)

# Definišemo loss funkciju i optimizator
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Trenira model
num_epochs = 100
loss = None
start_time = time.time()  # Početno vreme treniranja

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % eval_after == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()  # Krajnje vreme treniranja
training_duration = end_time - start_time  # Trajanje treniranja u sekundama

# Evaluacija modela
model.eval()
with torch.no_grad():
    predicted_stock_price = model(X_test)
    predicted_stock_price = mms.inverse_transform(predicted_stock_price.numpy())
    y_test = mms.inverse_transform(y_test.numpy())

# Prikazivanje rezultata
train = df[["Close/Last"]][:training_data_len]
valid = df[["Close/Last"]][training_data_len:]
valid['Predictions'] = predicted_stock_price
plt.figure(figsize=(10, 5))
plt.title('predicted vs true value')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Gold Close Price (USD)', fontsize=12)
plt.plot(train['Close/Last'])
plt.plot(valid[['Close/Last', 'Predictions']])

plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.savefig(f"data/figures/fig_{test_num}.png")
plt.show()

with open('data/results.txt', 'a') as f:
    f.write(f"{test_num:<8}, {lookback:<8}, {epoch+1:<5}, {hidden_size:<11}, {eval_after:<10},{training_duration/60:<15.2f}, {loss:<6f}\n")
