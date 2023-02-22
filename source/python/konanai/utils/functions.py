from ..session import Session
from ..dataloader import DataLoader
from ..nn import Module, Loss
from ..optim import Optimizer


def basic_train(dataloader: DataLoader, model: Module, loss_fn: Loss, optimizer: Optimizer, batch_report: int) -> None:
    """
    Train neural network model.
    This function update gradient.

    [1] DataLoader: Check Dataset inside Dataloader.
    [2] Model: Check each layer information and sub logic (input, output, activation, flatten, dense, add etc...)
    [3] Loss function: Check appropriate loss function.
    [4] Optimizer: Check appropriate Optimizer (arguments, etc...)

    Parameters
    ----------
    `dataloader`: DataLoader
        This parameter serve batch data.
    `model`: Module
        This parameter include layer information, sub logic, for neural network training.
    `loss_fn`: Loss
        This parameter calculate loss about prediction with answer.
    `optimizer`: Optimizer
        This parameter control gradients update value.
    `batch_report`: int
        This parameter report result when batch count is same multiple.
    """

    try :
        data_count = len(dataloader.dataset)
        batch_count = len(dataloader)
        current = 0

        model.train()
        
        for batch_index, (X, y) in enumerate(dataloader):
            #### Original code ################################
            # pred = model(X)
            # loss = loss_fn(pred, y)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # batch_size = len(X)
            # current += batch_size

            # if batch_index == 0:
            #     loss = loss.item()
            #     print(f"loss: {loss:>.7f}  [{current:>5d}/{data_count:>5d}]")
                
            # if batch_index % batch_report == batch_report-1:
            #     loss_value = loss.item()
            #     print("loss: {:.7f}".format(loss_value), " [{:5d}/{:5d}]".format(current, data_count))
                
            # if batch_index == batch_count-1:
            #     loss_value = loss.item()
            #     print("loss: {:.7f}".format(loss_value), " [{:5d}/{:5d}]".format(data_count, data_count))
            ###################################################
            
            #### Fixed to new style ###########################
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss_fn.backward()
            optimizer.step()
            batch_size = len(X['#'])
            current += batch_size
            
            if (batch_index + 1) % batch_report == 0 or batch_index == batch_count - 1:
                proc_count = loss_fn.get_proc_count(pred, y) # 최근 처리에서 실제 loss, accur 계산에 사용된 항의 갯수를 알려준다
                if len(loss) > 1:
                    loss_descs = []
                    loss_sum = 0
                    for key in loss.keys():
                        loss_term = loss[key].item() / proc_count[key]
                        loss_sum += loss_term
                        loss_term_desc = f"{key}:{loss_term:>7f}"
                        loss_descs.append(loss_term_desc)
                    loss_desc = f"{loss_sum:>7f}" + "(" + ",".join(loss_descs) + ")"
                else:
                    loss_term = loss["#"].item() / proc_count["#"]
                    loss_desc = f"{loss_term:>7f}"

                print(f"loss: {loss_desc}  [{current:>5d}/{data_count:>5d}]")
            ###################################################

    except Exception as e :
        raise Exception("Unknown exception occurred.")


def basic_test_classify(dataloader: DataLoader, model: Module, loss_fn: Loss, is_print_log: bool, session: Session) -> float:
    """
    Test neural network model.
    This function do not update gradient.
    Return average loss value.

    Parameters
    ----------
    `dataloader`: DataLoader
        This parameter serve batch data.
    `model`: Module
        This parameter include layer information, sub logic, for neural network training.
    `loss_fn`: Loss
        This parameter calculate loss about prediction with answer.
    `is_print_log`: bool
        This parameter control print-log or ignore.
    `session`: Session
        This parameter serve to connect 'Deep Learning Framework Engine' for ignore update gradient.

    Returns
    -------
    float
    """

    try :
        data_count = len(dataloader.dataset)
        batch_count = len(dataloader)
        test_loss = 0
        test_acc = 0
        correct = 0

        #### (Added) Fixed to new style ###################
        losses_acc = 0
        accurs_acc = 0

        model.eval()
        session.set_no_grad()
        
        for idx, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            #### Original code ################################
            # test_loss += loss.item()
            
            # # test_acc += loss_fn.eval_accuracy(pred, y).item()   # Bug: Last accuracy is wrong.
            # if idx == batch_count - 1:
            #     # Temporary code until bugfix
            #     over_size = (dataloader.batch_size * batch_count) - data_count
            #     last_data_size = dataloader.batch_size - over_size
            #     correct += (pred.to_ndarray().argmax(1) == y.to_ndarray().flatten()).astype(int)[0:last_data_size].sum()
            # else:
            #     correct += (pred.to_ndarray().argmax(1) == y.to_ndarray().flatten()).astype(int).sum()
            ###################################################
            
            #### Fixed to new style ###########################
            accur = loss_fn.eval_accuracy(pred, y)
            proc_count = loss_fn.get_proc_count(pred, y) # 실제 loss, accur 계산에 사용된 항의 갯수를 알려준다

            if losses_acc == 0:
                losses_acc = {}
                accurs_acc = {}
                proc_total = {}
                for key in loss:
                    losses_acc[key] = 0
                    accurs_acc[key] = 0
                    proc_total[key] = 0
            for key in loss:
                losses_acc[key] = losses_acc[key] + loss[key].item()
                accurs_acc[key] = accurs_acc[key] + accur[key].item()
                proc_total[key] = proc_total[key] + proc_count[key]
            ###################################################
                
        session.unset_no_grad()
        
        #### Original code ################################
        # test_loss /= batch_count
        # # test_acc /= batch_count # Bug: Last accuracy is wrong.
        # correct /= data_count
        
        # if is_print_log is True:
        #     # print(f"Accuracy: {(100 * test_acc):>0.1f}%")   # Bug: Last accuracy is wrong.
        #     print(f"Accuracy: {(100 * correct):>0.1f}%")
        #     print(f"Avg loss: {test_loss:>.7f}")
        #
        # return test_loss
        ###################################################
        
        #### Fixed to new style ###########################
        if len(losses_acc) > 1:
            loss_descs = []
            acc_descs = []
            loss_sum = 0
            acc_sum = 0
            for key in losses_acc.keys():
                loss_term = losses_acc[key] / proc_total[key]
                loss_sum += loss_term
                loss_term_desc = f"{key}:{loss_term:>7f}"
                loss_descs.append(loss_term_desc)
                loss_desc = f"{loss_sum:>7f}" + "(" + ",".join(loss_descs) + ")"

                acc_term = accurs_acc[key] / proc_total[key]
                acc_sum += acc_term
                acc_term_desc = f"{key}:{acc_term:>7f}"
                acc_descs.append(acc_term_desc)
                acc_desc = f"{acc_sum:>7f}" + "(" + ",".join(acc_descs) + ")"
        else:
            loss_term = losses_acc["#"] / proc_total["#"]
            loss_desc = f"{loss_term:>7f}"

            acc_term = accurs_acc["#"] / proc_total["#"]
            acc_desc = f"{acc_term:>7f}"

        print(f"Test Result: Loss: {loss_desc}, Accuracy: {acc_desc}")
        
        return float(loss_desc)
        ###################################################

    except Exception as e :
        raise Exception("Unknown exception occurred.")
