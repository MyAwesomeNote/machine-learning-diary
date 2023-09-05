import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("CUDA is available?", torch.cuda.is_available())

dataset = pd.read_csv('../data/car_evaluation.csv')
dataset.head()

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind="pie", autopct="%0.05f%%",
                                   colors=["lightblue", "lightgreen", "orange", "pink"],
                                   explode=(0.05, 0.05, 0.05, 0.05))
# plt.show()

category_col = ["price", "maint", "doors", "persons", "lug_capacity", "safety"]
for category in category_col:
    dataset[category] = dataset[category].astype("category")

price = dataset["price"].cat.codes.values
maint = dataset["maint"].cat.codes.values
doors = dataset["doors"].cat.codes.values
persons = dataset["persons"].cat.codes.values
lug_capacity = dataset["lug_capacity"].cat.codes.values
safety = dataset["safety"].cat.codes.values
category_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
categorical_data = torch.tensor(category_data, dtype=torch.int64)

outputs = pd.get_dummies(dataset.output).values
outputs = torch.Tensor(outputs).flatten()
categorical_col_size = [len(dataset[column].cat.categories) for column in category_col]
categorical_embedding_size = [(col_size, min(50, (col_size + 1) // 2)) for col_size in categorical_col_size]
total_records = 1728
test_records = int(total_records * .2)
categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[total_records - test_records:total_records]
train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[total_records - test_records:total_records]


class Model(nn.Module):
    def __init__(self, embedding_size, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        all_layers = []
        num_embeddings = sum((nf for ni, nf in embedding_size))
        input_size = num_embeddings
        for layer in layers:
            all_layers.append(nn.Linear(input_size, layer))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(layer))
            all_layers.append(nn.Dropout(p))
            input_size = layer
        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        embeddings = []
        for index, embedding in enumerate(self.all_embeddings):
            embeddings.append(embedding(x_categorical[:, index]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x


model = Model(categorical_embedding_size, 4, [200, 100, 50], p=0.4)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

epochs = 500
aggregated_losses = []
train_outputs: torch.Tensor = train_outputs.to(device, dtype=torch.int64)

single_loss = None
i = 0

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data).to(device)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i % 25 == 1:
        print(f"epoch : {i:3} loss : {single_loss.item():10.8f}")

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

test_outputs = test_outputs.to(device, dtype=torch.int64)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_function(y_val, test_outputs)

print(f'Loss: {loss:.8f}')
y_val = np.argmax(y_val, axis=1)
print(confusion_matrix(test_outputs, y_val))
print(classification_report(test_outputs, y_val))
print(accuracy_score(test_outputs, y_val))
