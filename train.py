from model.model import Model
from dataset.datatset import createDataset
from losses import TS2VecLoss
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

def collate(batch):
    data = []
    labels = []
    for i in range (len(batch)):
        data.append(batch[i][0])
        labels.append(batch[i][1])
    return torch.cat(data, dim=1), torch.cat(labels)


class Trainer():

    def __init__(self, datapath, epochs, save_path, log_path, model = None, model_path = None, attention=False, attention_heads=None, device = 'cpu'):

        assert (not model and model_path) or (not model_path and model)
        assert (not attention and not attention_heads) or (attention and attention_heads)

        if model_path:
            if attention:
                self.model = Model(attention=attention, attention_heads=attention_heads)
                self.model = self.model.load_state_dict(model_path)
            else:
                self.model = Model()
                self.model = self.model.load_state_dict(model_path)
        else:
            self.model = model
        self.model = self.model.to(device)
        self.attention = attention
        self.datapath = datapath
        size = (1,) if 'sleep-cassette' in datapath else (7,)
        #self.eval_idxs = np.random.randint(len(os.listdir(datapath)) - 1, size=size) # last folder has the lowest amount of samples
        self.eval_idxs = []
        self.epochs = epochs
        self.save_path=save_path
        self.log_path = log_path
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3)
        self.loss = TS2VecLoss(0.3)
        self.device = device

    def pretrain(self, view_strat): #To be used with You Snooze You Win datatset

        eval_folders = [os.listdir(self.datapath)[i] for i in self.eval_idxs]
        train_folders = [folder for folder in os.listdir(self.datapath) if folder not in eval_folders]
        epoch_loss = []
        eval_loss = []
        train_acc = []
        eval_acc = []
        with open(self.log_path, 'w') as log:
            log.write('#'*25 + ' TRAINING REPORT You Snooze You Win Dataset ' + '#'*25 + '\n')
            log.write(f'Training with MPNN_GAT: {self.attention}\n')
            for i in tqdm(range(self.epochs), desc='Training...'):
                self.model.train()
                datas_loss=[]
                datas_acc = []
                for dataset in tqdm(train_folders, desc=f'Epoch {i}', leave=False):
                    dset = createDataset(os.path.join(self.datapath, dataset))
                    dloader = torch.utils.data.DataLoader(dset, batch_size = 1, collate_fn=collate)
                    ls = [] # batch loss
                    preds, gt = [], []
                    for batch in tqdm(dloader, desc='Dataloader...', leave=False):
                        samples, labels = batch
                        samples = samples.permute(1, 0, 2).to(self.device)
                        labels = labels.to(self.device)
                        chs = samples.shape[1]
                        
                        if view_strat == 'split':
                                views = [[samples[j, i, :].unsqueeze(1) for i in range(chs)] for j in range(samples.shape[0])]
                        else: 
                            permuted_indices = torch.randperm(chs)
                            sep = torch.randint(2, chs-2)
                            group1 = permuted_indices[:sep]
                            group2 = permuted_indices[sep:]

                            views = [samples[:, group1, :].mean(dim=1, keepdim=True), 
                                    samples[:, group2, :].mean(dim=1, keepdim=True)]
                            print(len(views), len(views[0]), len(views[0][0]))

                        embeddings = [self.model(torch.tensor(np.array(v)).permute(2, 0, 1)) for v in views]
                        loss = 0
                        for i in range(len(embeddings)):
                            for j in range(i + 1, len(embeddings)):
                                loss += self.loss(embeddings[i], embeddings[j])
                        for i in range(len(embeddings)):
                            embeddings[i] = embeddings[i].detach().numpy()
                        
                        final = torch.tensor(np.mean(embeddings, axis=1)).to(self.device)
                        res = self.model.classify(final)
                        res = torch.argmax(res, 1)
                        preds.append(res)
                        gt.append(labels)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        ls.append(loss.item())
                    
                    preds = np.concatenate(preds, axis=None)
                    gt = np.concatenate(gt, axis=None)
                    datas_loss.append(np.mean(ls).item())
                    datas_acc.append(balanced_accuracy_score(gt, preds))
                epoch_loss.append(np.mean(datas_loss).item())
                train_acc.append(np.mean(datas_acc).item())
                log.write(f'Epoch {i}: \n\tTraining Loss: {epoch_loss[-1]}\n\tTraining Accuracy: {train_acc[-1]}')
                self.model.eval()
                e_loss, e_acc, prec, rec, f = self.eval(eval_folders, i, view_strat)
                
                eval_loss.append(e_loss)
                eval_acc.append(e_acc)
                log.write(f'\n\n\tValid Loss: {e_loss}\n\tValid Accuracy: {e_acc}')
                log.write(f'\n\tValid Precision: {prec}\n\tValid Recall: {rec}\n\tValid F1-score: {f}')
        
        torch.save(self.model.state_dict(), self.save_path)

        return epoch_loss, train_acc, eval_loss, eval_acc 

    def eval(self, eval_folders, i, view_strat):
        datas_loss=[]
        datas_acc = []
        datas_prec = []
        datas_rec = []
        datas_f = []
        for dataset in tqdm(eval_folders, desc=f'Evaluation after epoch {i}...', leave=False):
            with torch.no_grad():
                dset = createDataset(os.path.join(self.datapath, dataset))
                dloader = torch.utils.data.DataLoader(dset, batch_size = 1, collate_fn=collate)
                ls = [] # batch loss
                preds, gt = [], []
                for batch in tqdm(dloader, leave=False):
                    samples, labels = batch
                    samples = samples.permute(1, 0, 2).to(self.device)
                    labels = labels.to(self.device)
                    chs = samples.shape[1]
                    if view_strat == 'split':
                        views = [samples[:, i, :].reshape(1, -1) for i in range(chs)]
                    else: 
                        permuted_indices = torch.randperm(chs)
                        sep = torch.randint(2, chs-2)
                        group1 = permuted_indices[:sep]
                        group2 = permuted_indices[sep:]

                        views = [samples[:, group1, :].mean(dim=0, keepdim=True), 
                                samples[:, group2, :].mean(dim=0, keepdim=True)]

                    embeddings = [self.model(v) for v in views]
                    loss = 0
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            loss += self.loss(embeddings[i], embeddings[j])
                    ls.append(loss.item())
                    for i in range(len(embeddings)):
                        embeddings[i] = embeddings[i].detach().numpy()
                    final = torch.tensor(np.mean(embeddings, axis=1)).to(self.device)
                    preds.append(self.model.classify(final))
                    gt.append(labels)

                datas_loss.append(np.mean(ls).item())
                preds = np.concatenate(preds, axis=None)
                gt = np.concatenate(gt, axis=None)
                datas_acc.append(balanced_accuracy_score(gt, preds))
                prec, rec, f, _ = precision_recall_fscore_support(gt, preds)
                datas_prec.append(prec)
                datas_rec.append(rec)
                datas_f.append(f)


        return np.mean(datas_loss).item(), np.mean(datas_acc).item(), np.mean(datas_prec).item(), np.mean(datas_rec).item(), np.mean(datas_f).item()
      
    def plot(self, train_loss, valid_loss, train_acc, valid_acc):
        fig, (loss, acc) = plt.subplots(2, 1, sharex=True)

        fig.set_figwidth(15)
        fig.set_figheight(5)

        loss.set_title('Losses')
        loss.plot(train_loss, label='Training Loss')
        loss.plot(valid_loss, label='Evaluation Loss')
        loss.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        acc.set_title('Accuracies')
        acc.plot(train_acc, label='Training Accuracy')
        acc.plot(valid_acc, label='Evaluation Accuracy')
        acc.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        
        fig.savefig(f'./figures/plots.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close() 

    def finetune():
        pass

if __name__ == '__main__':
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(f'Using {device}')
    t=Trainer('data\Dataset\You snooze you win', 20, 'model_mpnn_conv.pt', 'model_mpnn_conv.log.txt', Model(), device='cpu')
    t.pretrain('split')