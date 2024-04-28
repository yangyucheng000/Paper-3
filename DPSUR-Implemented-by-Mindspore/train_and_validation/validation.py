from mindspore import Tensor
import mindspore.ops as ops
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def validation(model, test_loader):
    model.set_train(False)
    num_examples = 0
    test_loss = 0
    correct = 0

    for data, target in test_loader.create_tuple_iterator():
        data = Tensor(data, mstype.float32)
        target = Tensor(target, mstype.int32)
        target = ops.expand_dims(target, 0)
        logits = model(data)
        output = logits.logits if hasattr(logits, 'logits') else logits

        loss = nn.loss.CrossEntropyLoss()(output, target)
        test_loss += loss.asnumpy()

        pred = ops.Argmax(axis=1)(output)
        correct += (pred == target).sum().astype('float32').asnumpy()
        num_examples += data.shape[0]

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples
    return test_loss, test_acc
