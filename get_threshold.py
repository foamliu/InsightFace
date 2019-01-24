def naive():
    angles_file = 'data/angles.txt'
    print('Calculating threshold...')

    with open(angles_file, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'angle': angle, 'type': type})

    min_error = 6000
    min_threshold = 0

    for d in data:
        threshold = d['angle']
        type1 = len([s for s in data if s['angle'] <= threshold and s['type'] == 0])
        type2 = len([s for s in data if s['angle'] > threshold and s['type'] == 1])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    print(min_error, min_threshold)


def lr():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable

    angles_file = 'data/angles.txt'
    print('Calculating threshold...')

    with open(angles_file, 'r') as file:
        lines = file.readlines()

    X = np.zeros((6000, 1), np.float)
    T = np.zeros((6000, 1), np.long)

    for i, line in enumerate(lines):
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        X[i] = angle
        T[i] = type

    x_data = Variable(torch.Tensor(X))
    y_data = Variable(torch.Tensor(T))

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(1, 1)  # 2 in and 1 out

        def forward(self, x):
            y_pred = F.sigmoid(self.linear(x))
            return y_pred

    # Our model
    model = Model()

    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Training loop
    for epoch in range(1000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_data)

        # Compute and print loss
        loss = criterion(y_pred, y_data)
        print(epoch, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for f in model.parameters():
        print('data is')
        print(f.data)
        print(f.grad)

    w = list(model.parameters())
    w0 = w[0].data.numpy()
    w1 = w[1].data.numpy()


if __name__ == "__main__":
    lr()
