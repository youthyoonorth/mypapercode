import torch.optim as optim
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from dataloader_3D import Dataloader_3D
from model_icc import Model_3D # Baseline 2: cascaded LSTM (ICC)
import sys

# model_evaluation
def eval(model, loader, device, total, m):
    # input: trained model, test data loader, total (number of total points) and m (points to be predicted between two times of beam training)
    # output: accuracy, losses and normalized beamforming gain
    # reset dataloader
    loader.reset()
    # loss function
    criterion = nn.CrossEntropyLoss()
    # judge whether dataset is finished
    done = False
    # counting accurate prediction
    P = 0
    # counting inaccurate prediction
    N = 0
    # normalized beamforming gain, 10: beam training number, m: points to be predicted between two times of beam training
    BL = np.zeros((10, m))
    # running loss
    running_loss = 0
    # count batch number
    batch_num = 0
    # count predicted instants between two times of beam training
    # m = [0.2, 0.4, 0.6, 0.8]
    d_sery = np.linspace(0.2 - 1e-8, 0.8 - 1e-8, num = 4)

    # evaluate validation set
    while not done:
        # read files
        # channels: sequence of mmWave beam training received signal vectors
        # labels: sequence of optimal mmWave beam indices
        # beam power: sequence of mmWave beam training received signal power vectors
        channels, labels, beam_power, done, count = loader.next_batch()
        labels = labels.to(torch.int64)
        # select data for beam training
        channel_train = channels[:, :, 0 : total : (m + 1), :]
        if count == True:
            batch_num += 1
            # predicted results
            # out_tensor.shape: pre_points * length * batch_size * num_of_beam
            out_tensor = model(channel_train, pre_points = 4)
            # calculate loss function
            loss = 0
            # for all predictions after 10 beam trainings
            for loss_count in range(10):
                # for all predictions between two times of beam training
                for d_count in range(m):
                    t_d = np.abs(d_count * (1 / (m + 1) ) + (1 / (m + 1) ) - d_sery)
                    min_location = np.argmin(t_d)
                    loss += criterion(torch.squeeze(out_tensor[min_location, loss_count, :, :]),
                                      labels[:, loss_count * (m + 1) + d_count + 1])

            # output
            out_tensor_np = out_tensor.cpu().detach().numpy()
            # optimal beam index label
            gt_labels = labels.cpu().detach().numpy()
            gt_labels = np.float32(gt_labels)
            gt_labels = gt_labels.transpose(1, 0)
            # beam amplitude label
            beam_power = beam_power.cpu().detach().numpy()
            beam_power = beam_power.transpose(1, 0, 2)

            out_shape = gt_labels.shape
            for i in range(out_shape[0]):
                # i = (10 beam trainings + 10 beam trainings X m predictions, in time order)
                for j in range(out_shape[1]):
                    # j from 0 to batch_size - 1
                    # the instants of beam training will not be predicted
                    if i % (m + 1) != 0:
                        # number of beam trainings
                        loss_count = i // (m + 1)
                        # time offset between two times of beam training
                        d_count = i % (m + 1) - 1
                        # select the nearest predicted instant
                        t_d = np.abs(d_count * (1/( m + 1)) + (1/(m + 1)) - d_sery)
                        min_location = np.argmin(t_d)
                        # select the predicted result
                        train_ans = np.squeeze(out_tensor_np[min_location, loss_count, j, :])
                        train_index = np.argmax(train_ans)
                        # counting accurate and inaccurate prediction
                        if train_index == gt_labels[i, j]:
                            P = P + 1
                        else:
                            N = N + 1
                        # counting normalized beamforming gain
                        BL[loss_count, d_count] = BL[loss_count, d_count] + (beam_power[i, j, train_index] / max(
                            beam_power[i, j, :])) ** 2
            running_loss += loss.data.cpu()
    # average accuracy
    acur = float(P) / (P + N)
    # average loss
    losses = running_loss / batch_num
    # average beam power loss
    BL = BL / batch_num / 32
    # print results
    print("Accuracy: %.3f" % (acur))
    print("Loss: %.3f" % (losses))
    print("Beam power loss:")
    print(BL.T)
    return acur, losses, BL


# main function for model training and evaluation
# output: accuracy, losses and normalized beamforming gain
def main():
    # first loop for different velocities
    for velocity in [5, 10, 15, 20, 25, 30]:
        # save corresponding information
        print("velocity:", velocity)
        version_name = 'ICC_3CNN_1LSTM_160epoch_m=4_v1'
        # a = 0.2v m/s^2
        info = 'WCL_v' + str(velocity) + '_a' + str(velocity * 0.2) + '_25dBm_' + version_name
        print(info)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        t = 5  # training time
        epoch = 160 # training epoch
        batch_size = 32 # batch size
        total = 100 # total points
        m = 9 # the number of points to be predicted
        print('batch_size:%d' % (batch_size))

        path1 = '../../dataset/' + str(velocity) + '_train'
        path2 = '../../dataset/' + str(velocity) + '_test'
        loader = Dataloader_3D(path=path1, batch_size=batch_size, device=device)
        eval_loader = Dataloader_3D(path=path2, batch_size=batch_size, device=device)

        criterion = nn.CrossEntropyLoss()

        acur_train = []
        acur_eval = []
        loss_eval = []
        # length * prepoints * epoch * times_of_train
        BL_eval = np.zeros((10, m, epoch, t))  # normalized beamforming gain
        loss_train = []
        BL_train = []

        # first loop for training runnings
        for tt in range(t):
            print('Train %d times' % (tt))
            # learning rate
            lr = 0.00003
            model = Model_3D()
            model.to(device)
            # Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))
            # learning rate adaptive decay
            lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                                  verbose=True, threshold=0.0001,
                                                                  threshold_mode='rel', cooldown=0, min_lr=0.0000001,
                                                                  eps=1e-08)
            for name, param in model.named_parameters():
                print('Name:', name, 'Size:', param.size())

            # second loop for training times
            for e in range(epoch):
                print('Train %d epoch' % (e))
                loader.reset()
                eval_loader.reset()
                done = False
                running_loss = 0
                batch_num = 0
                while not done:
                    # read files
                    # channels: sequence of mmWave beam training received signal vectors
                    # labels: sequence of optimal mmWave beam indices
                    # beam power: sequence of mmWave beam training received signal power vectors
                    channels, labels, beam_power, done, count = loader.next_batch()
                    # select data for beam training
                    # batch_size * 2 * length * num of beam
                    channel_train = channels[:, :, 0 : total : (m + 1), :]
                    labels = labels.to(torch.int64)
                    if count == True:
                        batch_num += 1
                        # predicted results
                        # out_tensor.shape: pre_points * length * batch_size * num_of_beam
                        out_tensor = model(channel_train, pre_points = 4)
                        loss = 0
                        d_sery = np.linspace(0.2 - 1e-8, 0.8 - 1e-8, num = 4)
                        # m = 4, d_sery = [0.2 0.4 0.6 0.8];
                        # for all predictions after 10 beam trainings
                        for loss_count in range(10):
                            # time offset between two times of beam training
                            for d_count in range(m):
                                t_d = np.abs(d_count * ( 1 / (m + 1)) + ( 1 / (m + 1)) - d_sery)
                                # min_location is the closest time stamp
                                min_location = np.argmin(t_d)
                                # calculate prediction loss
                                loss += criterion(torch.squeeze(out_tensor[min_location, loss_count, :, :]),
                                                  labels[:, loss_count * (m + 1) + d_count + 1])
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                losses = running_loss / batch_num
                print('[%d] loss: %.3f' %
                      (e + 1, losses))
                loss_train.append(losses)
                model.eval()
                print('the evaling set:')
                if 1:
                    acur, losses, BL = eval(model, eval_loader, device, total, m)
                    acur_eval.append(acur)
                    loss_eval.append(losses)
                    BL_eval[:, :, e, tt] = np.squeeze(BL)
                # learning rate decay
                lr_decay.step(losses)
                # train mode, where dropout is on
                model.train()

                mat_name = info + '.mat'
                sio.savemat(mat_name, {'acur_train': acur_train,
                                       'acur_eval': acur_eval,
                                       'loss_train': loss_train, 'loss_eval': loss_eval,
                                       'BL_train': BL_train, 'BL_eval': BL_eval})



if __name__ == '__main__':
    main()