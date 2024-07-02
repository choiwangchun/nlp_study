import torch.nn as nn
import torch.optim as optim
import torch

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        # nn.Module에 있는 초기값 가져오고 입력특성 크기 만들기
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        #퍼셉트론 정방향 계산
        return torch.sigmoid(self.fc1(X_in)).squeeze()



input_dim = 2
lr = 0.001



perceptron = Perceptron(input_dim=input_dim)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)

#에폭 반복
for epoch in range(n_epochs):
    #배치 반복
    for batch_i in range(n_batch_i):
        #데이터 가져오기
        x_data, y_target = get_toy_data(batch_size)
        #그레디언트 초기화
        perceptron.zero_grad()
        #y값 예측
        y_pred = perceptron(x_data, apply_sigmoid=True)
        #손실함수 계산
        loss = bce_loss(y_pred, y_target)
        #손실신호 역전파
        loss.backward()
        #옵티마이저로 업데이트
        optimizer.step()
