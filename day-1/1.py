import torch
import torch.nn as nn
import torch.optim as optim

x=torch.randn(100,20)
y=torch.randint(0,2,(100,))

class SimpleMlP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(20,64)
        self.fc2=nn.Linear(64,2)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x

model=SimpleMlP().cuda()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10000):
    optimizer.zero_grad()
    outputs=model(x.cuda())
    loss=criterion(outputs,y.cuda())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")