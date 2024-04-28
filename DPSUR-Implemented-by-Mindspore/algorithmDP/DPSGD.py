from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from train_and_validation.train_with_dp import  DPtrain
from train_and_validation.validation import validation
import data.util.dataloader as dl

def DPSGD(train_data, test_data, model,l2_norm_clip, learning_rate, batch_size, epsilon_budget, delta,sigma,device):


    test_dl = dl.load_test_data(is_dpsur = False, batch_size = batch_size)
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    iter = 1
    epsilon = 0.
    best_test_acc=0.
    while epsilon < epsilon_budget:

        epsilon, best_alpha = apply_dp_sgd_analysis(batch_size/60000, sigma, iter, orders, delta) #comupte privacy cost
        train_dl = dl.load_train_data_minibatch(minibatch_size= batch_size, iterations = 1)
        dp_train = DPtrain(model, train_dl, l2_norm_clip, sigma, batch_size, learning_rate, device)
        train_loss, train_accuracy = dp_train.train_with_dp()

        test_loss, test_accuracy = validation(model, test_dl)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter

        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter += 1

    print("------finished ------")
    return test_accuracy,iter,best_test_acc,best_iter