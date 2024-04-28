import numpy as np
import os
def dividing_validation_set(train_data, validation_num):
    mnist_data = np.load('data\\mnist\\mnist.npz')

    data, label = mnist_data['x_train'], mnist_data['y_train']
    test_data, test_label = mnist_data['x_test'], mnist_data['y_test']
    train_indices = np.arange(len(data) - validation_num)
    valid_indices = np.arange(len(data) - validation_num, len(data))

    train_data, train_label = data[train_indices], label[train_indices]
    valid_data, valid_label = data[valid_indices], label[valid_indices]

    train_data = train_data.astype(np.float32) / 255.0
    train_label = train_label.astype(np.int32)
    valid_data = valid_data.astype(np.float32) / 255.0
    valid_label = valid_label.astype(np.int32)
    test_data = test_data.astype(np.float32) / 255.0
    test_label = test_label.astype(np.int32)

    train_data = np.expand_dims(train_data, axis=1)
    valid_data = np.expand_dims(valid_data, axis=1)
    test_data = np.expand_dims(test_data, axis = 1)

    train_dataset = (train_data, train_label)
    valid_dataset = (valid_data, valid_label)
    test_dataset = (test_data, test_label)

    return train_dataset, valid_dataset, test_dataset
