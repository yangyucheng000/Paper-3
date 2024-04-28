import mindspore.nn as nn

def train(model, train_loader, optimizer):
    train_loss = 0.0
    train_acc = 0.0

    for id, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = nn.SoftmaxCrossEntropyWithLogits()(output, target)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

    return train_loss, train_acc
