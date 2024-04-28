import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from utils.dp_optimizer import DPoptimizer
import sys


class DPtrain():
    def __init__(self, model, train_loader, l2_norm_clip, dp_sigma, batch_size, learning_rate, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.loss_fn = nn.loss.CrossEntropyLoss()

        self.optimizer = DPoptimizer(
                        l2_norm_clip = l2_norm_clip,
                        noise_multiplier = dp_sigma,
                        minibatch_size = batch_size,
                        microbatch_size = 1,
                        params = model.trainable_params(),
                        learning_rate = learning_rate,
                    )


    def forward_fn(self, data, label):
        logits = self.model(data)
        label = label.astype(mindspore.int32)
        loss = self.loss_fn(logits, label)
        return loss, logits

    def train_step_dpsgd(self, data, label):
        grad_fn = mindspore.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        (loss, __), grads = grad_fn(data, label)
        return loss, grads

    def train_with_dp(self):

        sys.stdout.flush()
        self.model.set_train()

        max_local_epochs = 1
        for step in range(max_local_epochs):
            gradients_list = []
            losses_list = []
            for data in self.train_loader.create_dict_iterator():
                x = data['image']
                y = data['label']
                x_single = x
                y_single = ops.expand_dims(y, 0)
                loss, grad = self.train_step_dpsgd(x_single, y_single)
                gradients_list.append(grad)
                losses_list.append(loss)
            self.optimizer(gradients_list)
            self.loss_batch_avg = sum(losses_list) / len(losses_list)

        return 0., 0.
