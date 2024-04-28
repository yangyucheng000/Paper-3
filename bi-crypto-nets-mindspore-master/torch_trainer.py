import os
import random
import numpy as np
import mindspore
from mindspore import context
from mindspore import nn, context, Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.common.initializer import HeNormal, Constant
from mindspore.ops import operations as ops

from utils import progress_bar

def seed_mindspore(seed=100) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    return seed

from mindspore import load_checkpoint, save_checkpoint

class MindSpore_Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler=None, ckpt_name=None, ckpt_dir='./checkpoint', load_best=False, clip_norm=0.):
        self.net = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.clip_norm = clip_norm
        self.device = context.get_context('device_target')

        self.best_acc = 0  # best test accuracy
        self.best_loss = float('inf')

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # Checkpoint file name
        if ckpt_name is None:
            self.ckpt_name = '%s.ckpt' % self.net.network_name
        else:
            self.ckpt_name = '%s.ckpt' % ckpt_name

        # Load the best model if required
        if load_best:
            self.load_best()

    def save_checkpoint(self, epoch):
        save_checkpoint(self.net, os.path.join(self.ckpt_dir, self.ckpt_name.format(epoch=epoch)))

    def load_best(self):
        ckpt_file = os.path.join(self.ckpt_dir, self.ckpt_name)
        if os.path.exists(ckpt_file):
            param_dict = mindspore.load_checkpoint(ckpt_file)
            mindspore.load_param_into_net(self.net, param_dict)

    def train_epoch(self, train_loader, epoch):
        self.net.set_train()
        total_samples = 0
        correct_samples = 0
        total_loss = 0

        for batch_data in train_loader.create_tuple_iterator():
            inputs, targets = batch_data
            outputs = self.net(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_norm > 1e-6:
                nn.clip_grads(self.net.trainable_params(), clip_value=self.clip_norm)
            self.optimizer.step()

            # Calculate accuracy
            total_samples += targets.shape[0]
            correct_samples += (outputs.argmax(axis=1) == targets).sum().astype('float')
            total_loss += loss

        avg_loss = total_loss / total_samples
        accuracy = correct_samples / total_samples
        print(f'Epoch {epoch} - Loss: {avg_loss}, Accuracy: {accuracy}')

    def test(self, test_loader, epoch):
        self.net.set_eval()
        total_samples = 0
        correct_samples = 0

        for batch_data in test_loader.create_tuple_iterator():
            inputs, targets = batch_data
            outputs = self.net(inputs)

            # Calculate accuracy
            total_samples += targets.shape[0]
            correct_samples += (outputs.argmax(axis=1) == targets).sum().astype('float')

        accuracy = correct_samples / total_samples
        print(f'Test - Epoch {epoch} - Accuracy: {accuracy}')
        return accuracy

    def fit(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch)
            accuracy = self.test(test_loader, epoch)

            # Save best model
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.save_checkpoint(epoch)

            if self.scheduler:
                self.scheduler.step()


class KD_MindSpore_Trainer(MindSpore_Trainer):
    def __init__(self, model, teacher, loss_fn, distillation_loss_fn, optimizer, scheduler=None, ckpt_name=None, ckpt_dir='./checkpoint', load_best=False, alpha=0.1, temperature=4, clip_norm=0.):
        super().__init__(model, loss_fn, optimizer, scheduler, ckpt_name, ckpt_dir, load_best, clip_norm)
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_loss_fn = distillation_loss_fn

        self.teacher.set_train(False)  # Teacher model in evaluation mode
        for param in self.teacher.get_parameters():
            param.requires_grad = False

    def train_epoch(self, train_loader, epoch):
        self.net.set_train()
        total_samples = 0
        correct_samples = 0
        total_loss = 0

        softmax = ops.Softmax(axis=1)
        for batch_data in train_loader.create_tuple_iterator():
            (inputs0, inputs1), x, targets = batch_data

            # Forward pass in teacher and student models
            teacher_outputs, _ = self.teacher(x, True)
            s_fig, student_outputs = self.net([inputs0, inputs1], True)

            # Compute the distillation loss
            distillation_loss = self.distillation_loss_fn(softmax(teacher_outputs / self.temperature), softmax(s_fig / self.temperature))

            # Combine with the standard loss
            student_loss = self.loss_fn(student_outputs, targets)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss * (self.temperature ** 2)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_norm > 1e-6:
                nn.clip_grads(self.net.trainable_params(), clip_value=self.clip_norm)
            self.optimizer.step()

            # Calculate accuracy
            total_samples += targets.shape[0]
            correct_samples += (student_outputs.argmax(axis=1) == targets).sum().astype('float')
            total_loss += loss

        avg_loss = total_loss / total_samples
        accuracy = correct_samples / total_samples
        print(f'Epoch {epoch} - Loss: {avg_loss}, Accuracy: {accuracy}')


class KD_MindSpore_Trainer_2(KD_MindSpore_Trainer):
    def __init__(self, model, teacher, loss_fn, distillation_loss_fn, optimizer, scheduler=None, ckpt_name=None, ckpt_dir='./checkpoint', load_best=False, alpha=0.1, temperature=4, clip_norm=0.):
        super().__init__(model, teacher, loss_fn, distillation_loss_fn, optimizer, scheduler, ckpt_name, ckpt_dir, load_best, alpha, temperature, clip_norm)

    def train_epoch(self, train_loader, epoch):
        self.net.set_train()
        total_samples = 0
        correct_samples = 0
        total_loss = 0

        softmax = ops.Softmax(axis=1)
        for batch_data in train_loader.create_tuple_iterator():
            ((inputs0, inputs1), x, targets) = batch_data

            # Forward pass in teacher and student models
            _, teacher_outputs = self.teacher(x, True)
            _, student_outputs = self.net([inputs0, inputs1], True)

            # Compute the distillation loss
            distillation_loss = self.temperature ** 2 * self.distillation_loss_fn(
                softmax(teacher_outputs / self.temperature),
                softmax(student_outputs / self.temperature)
            )

            # Combine with the student's loss on true labels
            student_loss = self.loss_fn(student_outputs, targets)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_norm > 1e-6:
                nn.clip_grads(self.net.trainable_params(), clip_value=self.clip_norm)
            self.optimizer.step()

            # Calculate accuracy
            total_samples += targets.shape[0]
            correct_samples += (student_outputs.argmax(axis=1) == targets).sum().astype('float')
            total_loss += loss

        avg_loss = total_loss / total_samples
        accuracy = correct_samples / total_samples
        print(f'Epoch {epoch} - Loss: {avg_loss}, Accuracy: {accuracy}')
