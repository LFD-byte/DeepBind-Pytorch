from utils import *
from model import ConvNet
from pre_process import *
import torch.nn.functional as F
from config import *


# Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

chipseq = Chip('data/encode/ELK1_GM12878_ELK1_(1277-1)_Stanford_AC.seq.gz')

train1, valid1, train2, valid2, train3, valid3, alldataset = chipseq.openFile()

train1_dataset = chipseq_dataset(train1)
train2_dataset = chipseq_dataset(train2)
train3_dataset = chipseq_dataset(train3)
valid1_dataset = chipseq_dataset(valid1)
valid2_dataset = chipseq_dataset(valid2)
valid3_dataset = chipseq_dataset(valid3)
alldataset_dataset = chipseq_dataset(alldataset)

batchSize = 64
if reverse_mode:
    train_loader1 = DataLoader(dataset=train1_dataset, batch_size=batchSize, shuffle=False)
    train_loader2 = DataLoader(dataset=train2_dataset, batch_size=batchSize, shuffle=False)
    train_loader3 = DataLoader(dataset=train3_dataset, batch_size=batchSize, shuffle=False)
    valid1_loader = DataLoader(dataset=valid1_dataset, batch_size=batchSize, shuffle=False)
    valid2_loader = DataLoader(dataset=valid2_dataset, batch_size=batchSize, shuffle=False)
    valid3_loader = DataLoader(dataset=valid3_dataset, batch_size=batchSize, shuffle=False)
    alldataset_loader = DataLoader(dataset=alldataset_dataset, batch_size=batchSize, shuffle=False)
else:
    train_loader1 = DataLoader(dataset=train1_dataset, batch_size=batchSize, shuffle=True)
    train_loader2 = DataLoader(dataset=train2_dataset, batch_size=batchSize, shuffle=True)
    train_loader3 = DataLoader(dataset=train3_dataset, batch_size=batchSize, shuffle=True)
    valid1_loader = DataLoader(dataset=valid1_dataset, batch_size=batchSize, shuffle=False)
    valid2_loader = DataLoader(dataset=valid2_dataset, batch_size=batchSize, shuffle=False)
    valid3_loader = DataLoader(dataset=valid3_dataset, batch_size=batchSize, shuffle=False)
    alldataset_loader = DataLoader(dataset=alldataset_dataset, batch_size=batchSize, shuffle=False)

train_dataloader = [train_loader1, train_loader2, train_loader3]
valid_dataloader = [valid1_loader, valid2_loader, valid3_loader]

best_AUC = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)
# device='cpu'
learning_steps_list = [4000, 8000, 12000, 16000, 20000]
for number in range(num_epochs):

    pool_List = ['max', 'maxavg']
    random_pool = random.choice(pool_List)

    neuType_list = ['hidden', 'nohidden']
    random_neuType = random.choice(neuType_list)
    dropoutList = [0.5, 0.75, 1.0]

    dropprob = random.choice(dropoutList)

    learning_rate = logsampler(0.0005, 0.05)

    momentum_rate = sqrtsampler(0.95, 0.99)

    sigmaConv = logsampler(10 ** -7, 10 ** -3)

    sigmaNeu = logsampler(10 ** -5, 10 ** -2)
    beta1 = logsampler(10 ** -15, 10 ** -3)
    beta2 = logsampler(10 ** -10, 10 ** -3)
    beta3 = logsampler(10 ** -10, 10 ** -3)

    model_auc = [[], [], []]
    for kk in range(3):
        model = ConvNet(16, 24, random_pool, random_neuType, 'training', dropprob, learning_rate, momentum_rate,
                        sigmaConv, sigmaNeu, beta1, beta2, beta3, reverse_complemet_mode=reverse_mode).to(device)
        if random_neuType == 'nohidden':
            optimizer = torch.optim.SGD([model.wConv, model.wRect, model.wNeu, model.wNeuBias], lr=model.learning_rate,
                                        momentum=model.momentum_rate, nesterov=True)

        else:
            optimizer = torch.optim.SGD(
                [model.wConv, model.wRect, model.wNeu, model.wNeuBias, model.wHidden, model.wHiddenBias],
                lr=model.learning_rate, momentum=model.momentum_rate, nesterov=True)

        train_loader = train_dataloader[kk]
        valid_loader = valid_dataloader[kk]
        learning_steps = 0
        while learning_steps <= 20000:
            model.mode = 'training'
            auc = []
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                if model.reverse_complemet_mode:
                    target_2 = torch.randn(int(target.shape[0] / 2), 1)
                    for i in range(target_2.shape[0]):
                        target_2[i] = target[2 * i]
                    target = target_2.to(device)

                # Forward pass
                output = model(data)
                if model.neuType == 'nohidden':
                    loss = F.binary_cross_entropy(torch.sigmoid(output),
                                                  target) + model.beta1 * model.wConv.norm() + model.beta3 * model.wNeu.norm()

                else:
                    loss = F.binary_cross_entropy(torch.sigmoid(output),
                                                  target) + model.beta1 * model.wConv.norm() + model.beta2 * model.wHidden.norm() + model.beta3 * model.wNeu.norm()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learning_steps += 1

                if learning_steps % 4000 == 0:

                    with torch.no_grad():
                        model.mode = 'test'
                        auc = []
                        for i, (data, target) in enumerate(valid_loader):
                            data = data.to(device)
                            target = target.to(device)
                            if model.reverse_complemet_mode:
                                target_2 = torch.randn(int(target.shape[0] / 2), 1)
                                for i in range(target_2.shape[0]):
                                    target_2[i] = target[2 * i]
                                target = target_2.to(device)
                            # Forward pass
                            output = model(data)
                            pred_sig = torch.sigmoid(output)
                            pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                            labels = target.cpu().numpy().reshape(output.shape[0])

                            auc.append(metrics.roc_auc_score(labels, pred))
                        #                         print(np.mean(auc))
                        model_auc[kk].append(np.mean(auc))
                        print('AUC performance when training fold number ', kk + 1, 'using ',
                              learning_steps_list[len(model_auc[kk]) - 1], 'learning steps = ', np.mean(auc))

    print('                   ##########################################               ')
    for n in range(5):
        AUC = (model_auc[0][n] + model_auc[1][n] + model_auc[2][n]) / 3
        # print(AUC)
        if AUC > best_AUC:
            best_AUC = AUC
            best_learning_steps = learning_steps_list[n]
            best_LearningRate = model.learning_rate
            best_LearningMomentum = model.momentum_rate
            best_neuType = model.neuType
            best_poolType = model.poolType
            best_sigmaConv = model.sigmaConv
            best_dropprob = model.dropprob
            best_sigmaNeu = model.sigmaNeu
            best_beta1 = model.beta1
            best_beta2 = model.beta2
            best_beta3 = model.beta3

print('best_poolType=', best_poolType)
print('best_neuType=', best_neuType)
print('best_AUC=', best_AUC)
print('best_learning_steps=', best_learning_steps)
print('best_LearningRate=', best_LearningRate)
print('best_LearningMomentum=', best_LearningMomentum)
print('best_sigmaConv=', best_sigmaConv)
print('best_dropprob=', best_dropprob)
print('best_sigmaNeu=', best_sigmaNeu)
print('best_beta1=', best_beta1)
print('best_beta2=', best_beta2)
print('best_beta3=', best_beta3)

best_hyperparameters = {'best_poolType': best_poolType, 'best_neuType': best_neuType,
                        'best_learning_steps': best_learning_steps, 'best_LearningRate': best_LearningRate,
                        'best_LearningMomentum': best_LearningMomentum, 'best_sigmaConv': best_sigmaConv,
                        'best_dropprob': best_dropprob,
                        'best_sigmaNeu': best_sigmaNeu, 'best_beta1': best_beta1, 'best_beta2': best_beta2,
                        'best_beta3': best_beta3}
torch.save(best_hyperparameters, 'best_hyperpamarameters.pth')