from model.model import Model
from dataset.datatset import createDataset
from losses import TS2VecLoss
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
import random

def collate(batch):
    data = []
    labels = []
    for i in range (len(batch)):
        data.append(batch[i][0])
        labels.append(batch[i][1])
    return torch.cat(data, dim=1), torch.cat(labels)


class Trainer():

    def __init__(self, datapath, epochs, save_path, log_path, model = None, model_path = None, attention=False, attention_heads=None, device = 'cpu', view_strat = 'split', finetune = False):

        assert (not model and model_path) or (not model_path and model)
        assert (not attention and not attention_heads) or (attention and attention_heads)
        self.model_path = model_path
        if model_path:
            if attention:
                self.model = Model(attention=attention, attention_heads=attention_heads, view_strat=view_strat, finetune=finetune)
            else:
                self.model = Model(view_strat=view_strat, finetune=finetune)
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            self.model = model
        self.model = self.model.to(device)
        self.attention = attention
        self.datapath = datapath
        size = (1,) if 'sleep-cassette' in datapath else (7,)
        self.eval_idxs = np.random.randint(len(os.listdir(datapath)) - 1, size=size) # last folder has the lowest amount of samples
        self.eval_folders = [os.listdir(self.datapath)[i] for i in self.eval_idxs]
        self.train_folders = [folder for folder in os.listdir(self.datapath) if folder not in self.eval_folders]
        self.epochs = epochs
        self.save_path=save_path
        self.log_path = log_path
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3)
        self.loss = TS2VecLoss(0.3)
        self.device = device

    def pretrain(self, view_strat): #To be used with You Snooze You Win dataset
        epoch_loss = []
        eval_loss = []
        train_acc = []
        eval_acc = []
        with open(self.log_path, 'w') as log:
            log.write('#'*25 + ' TRAINING REPORT You Snooze You Win Dataset ' + '#'*25 + '\n')
            log.write(f'Training with MPNN_GAT: {self.attention}\n')
            log.write(f'Training with view strategy: {view_strat}\n')
            for epoch in tqdm(range(self.epochs), desc='Training...'):
                self.model.train()
                datas_loss=[]
                datas_acc = []
                for dataset in tqdm(self.train_folders, desc=f'Epoch {epoch}', leave=False):
                    dset = createDataset(os.path.join(self.datapath, dataset))
                    dloader = torch.utils.data.DataLoader(dset, batch_size = 1, collate_fn=collate)
                    ls = [] # batch loss
                    preds, gt = [], []
                    for batch in tqdm(dloader, desc='Dataloader...', leave=False):
                        samples, labels = batch
                        choice = np.random.choice(np.arange(len(labels), step=1),replace=False, size=(len(labels)//4,))
                        choice.sort()
                        samples = samples.permute(1, 0, 2)[choice]
                        labels = labels[choice]
                        chs = samples.shape[1]
                        views = np.array([[samples[j, i, :].unsqueeze(1) for i in range(chs)] for j in range(samples.shape[0])])
                        embeddings = [self.model.encode(torch.tensor(v).permute(2, 0, 1).to(self.device)) for v in views]
                        embeddings = torch.stack(embeddings)
                        if view_strat == 'split':
                            num_views = embeddings.shape[1]
                            i, j = torch.triu_indices(num_views, num_views, offset=1)
                            losses = self.loss(embeddings[:, i, :, :], embeddings[:, j, :, :])
                            loss = losses.sum()
                            res = self.model.classify(embeddings.permute(0, 2, 3, 1)).argmax(dim=1)
                            
                        else: 
                            permuted_indices = torch.randperm(chs)
                            sep = torch.randint(low=2, high=chs-1, size=(1,))
                            group1 = permuted_indices[:sep]
                            group2 = permuted_indices[sep:]
                            v1, v2 = embeddings[:, group1, :, :], embeddings[:, group2, :, :]
                            embedding1 = self.model.message(v1.permute(0, 1, 3, 2)).mean(dim=1, keepdim=True)
                            embedding2 = self.model.message(v2.permute(0, 1, 3, 2)).mean(dim=1, keepdim=True)
                            loss = self.loss(embedding1, embedding2)
                            embeddings = self.model.mpnn.rout(embedding1, embedding2)
                            res = self.model.classify(embeddings.permute((1, 2, 3, 0))).argmax(dim=1)
                            
                        preds.append(res.cpu())
                        gt.append(labels.cpu())
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
                log.write(f'Epoch {epoch}: \n\tTraining Loss: {epoch_loss[-1]}\n\tTraining Accuracy: {train_acc[-1]}')
                self.model.eval()
                e_loss, e_acc, prec, rec, f = self.eval(self.eval_folders, epoch, view_strat)
                
                eval_loss.append(e_loss)
                eval_acc.append(e_acc)
                log.write(f'\n\tValid Loss: {e_loss}\n\tValid Accuracy: {e_acc}')
                log.write(f'\n\tValid Precision: {prec}\n\tValid Recall: {rec}\n\tValid F1-score: {f}\n')
        
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
                    choice = np.random.choice(np.arange(len(labels), step=1),replace=False, size=(len(labels)//4,))
                    choice.sort()
                    samples = samples.permute(1, 0, 2)[choice]
                    labels = labels[choice]
                    chs = samples.shape[1]
                    views = np.array([[samples[j, i, :].unsqueeze(1) for i in range(chs)] for j in range(samples.shape[0])])
                    embeddings = [self.model.encode(torch.tensor(v).permute(2, 0, 1).to(self.device)) for v in views]
                    embeddings = torch.stack(embeddings)
                    
                    if view_strat == 'split':
                        num_views = embeddings.shape[1]
                        i, j = torch.triu_indices(num_views, num_views, offset=1)
                        losses = self.loss(embeddings[:, i, :, :], embeddings[:, j, :, :])
                        loss = losses.sum()
                        res = self.model.classify(embeddings.permute(0, 2, 3, 1)).argmax(dim=1)
                        
                    else: 
                        permuted_indices = torch.randperm(chs)
                        sep = torch.randint(low=2, high=chs-1, size=(1,))
                        group1 = permuted_indices[:sep]
                        group2 = permuted_indices[sep:]
                        v1, v2 = embeddings[:, group1, :, :], embeddings[:, group2, :, :]
                        embedding1 = self.model.message(v1.permute(0, 1, 3, 2)).mean(dim=1, keepdim=True)
                        embedding2 = self.model.message(v2.permute(0, 1, 3, 2)).mean(dim=1, keepdim=True)
                        loss = self.loss(embedding1, embedding2)
                        embeddings = self.model.mpnn.rout(embedding1, embedding2)
                        res = self.model.classify(embeddings.permute((1, 2, 3, 0))).argmax(dim=1)
                    
                    ls.append(loss.item())
                    preds.append(res.cpu())
                    gt.append(labels.cpu())

                datas_loss.append(np.mean(ls).item())
                preds = np.concatenate(preds, axis=None)
                gt = np.concatenate(gt, axis=None)
                datas_acc.append(balanced_accuracy_score(gt, preds))
                prec, rec, f, _ = precision_recall_fscore_support(gt, preds)
                datas_prec.append(prec)
                datas_rec.append(rec)
                datas_f.append(f)


        return np.mean(datas_loss).item(), np.mean(datas_acc).item(), np.mean(datas_prec).item(), np.mean(datas_rec).item(), np.mean(datas_f).item()
      
    def plot(self, train_loss, valid_loss, train_acc, valid_acc, view_strat, finetune):
        fig, (loss, acc) = plt.subplots(2, 1, sharex=True)

        fig.set_figwidth(15)
        fig.set_figheight(5)

        loss.set_title('Losses' if valid_loss else 'Training Loss')
        loss.plot(train_loss, label='Training Loss')
        if valid_loss:
            loss.plot(valid_loss, label='Evaluation Loss')
            loss.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        acc.set_title('Accuracies' if valid_acc else 'Training accuracy')
        acc.plot(train_acc, label='Training Accuracy')
        if valid_acc:
            acc.plot(valid_acc, label='Evaluation Accuracy')
            acc.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        f = 'finetune' if finetune else 'pretraining'
        fig.savefig(f'./figures/plots_{view_strat}_{f}.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close() 

    def finetune(self, view_strat): #To be used with sleep-cassette dataset
        assert self.model_path 
        self.model.update_classifier(2, 5)
        self.model = self.model.to(self.device)
        epoch_loss = []
        train_acc = []
        with open(self.log_path, 'w') as log:
            log.write('#'*25 + ' FINETUNING REPORT Sleep-Cassette Dataset ' + '#'*25 + '\n')
            log.write(f'Training with MPNN_GAT: {self.attention}\n')
            log.write(f'Training with view strategy: {view_strat}\n')
            for epoch in tqdm(range(self.epochs), desc='Training...'):
                self.model.train()
                datas_loss=[]
                datas_acc = []
                for dataset in tqdm(self.train_folders, desc=f'Epoch {epoch}', leave=False):
                    dset = createDataset(os.path.join(self.datapath, dataset))
                    dloader = torch.utils.data.DataLoader(dset, batch_size = 1, collate_fn=collate)
                    ls = [] # batch loss
                    preds, gt = [], []
                    for batch in tqdm(dloader, desc='Dataloader...', leave=False):
                        samples, labels = batch
                        choice = np.random.choice(np.arange(len(labels), step=1),replace=False, size=(len(labels)//4,))
                        choice.sort()
                        samples = samples.permute(1, 0, 2)[choice]
                        labels = labels[choice]
                        chs = samples.shape[1]
                        views = np.array([[samples[j, i, :].unsqueeze(1) for i in range(chs)] for j in range(samples.shape[0])])
                        embeddings = [self.model.encode(torch.tensor(v).permute(2, 0, 1).to(self.device)) for v in views]
                        embeddings = torch.stack(embeddings)
                        if view_strat == 'split':
                            loss = self.loss(embeddings[:, 0, :, :], embeddings[:, 1, :, :])
                            res = self.model.classify(embeddings.permute(0, 2, 3, 1)).argmax(dim=1)
                            
                        else: 
                            v1, v2 = embeddings[:, 0, :, :].unsqueeze(1), embeddings[:, 1, :, :].unsqueeze(1)
                            embedding1 = self.model.message(v1.permute(0, 1, 3, 2)).mean(dim=1, keepdim=True)
                            embedding2 = self.model.message(v2.permute(0, 1, 3, 2)).mean(dim=1, keepdim=True)
                            loss = self.loss(embedding1, embedding2)
                            embeddings = self.model.mpnn.rout(embedding1, embedding2)
                            res = self.model.classify(embeddings.permute((1, 2, 3, 0)).to(device)).argmax(dim=1)
                            
                        preds.append(res.cpu())
                        gt.append(labels.cpu())
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
                log.write(f'Epoch {epoch}: \n\tTraining Loss: {epoch_loss[-1]}\n\tTraining Accuracy: {train_acc[-1]}\n')
            self.model.eval()
            e_loss, e_acc, prec, rec, f = self.eval(self.eval_folders, epoch, view_strat)
            
            log.write('#'*25 + ' Test Result ' + '#'*25 + '\n')
            log.write(f'\n\Test Loss: {e_loss}\n\Test Accuracy: {e_acc}')
            log.write(f'\n\Test Precision: {prec}\n\Test Recall: {rec}\n\tValid F1-score: {f}\n')
        
        torch.save(self.model.state_dict(), self.save_path)

        return epoch_loss, train_acc
        

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    seed = 3233
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    if not os.path.exists('./models') and not os.path.exists('./logs'):
        os.makedirs('./models')
        os.makedirs('./logs')
        
    t=Trainer('data\Dataset\You snooze you win', 20, './models/model_mpnn_conv_group.pt', './logs/model_mpnn_conv_group.log.txt', Model(6, view_strat='group'), device=device, view_strat='group')
    t_loss, t_acc, val_loss, val_acc = t.pretrain('group')
    t.plot(t_loss, val_loss, t_acc, val_acc, 'group', False)
    
    t=Trainer('data\Dataset\You snooze you win', 20, 'model_mpnn_conv_split.pt', 'model_mpnn_conv_split.log.txt', Model(6, view_strat='split'), device=device, view_strat='split')
    t_loss, t_acc, val_loss, val_acc = t.pretrain('split')
    t.plot(t_loss, val_loss, t_acc, val_acc, 'split', False)
    
    t = Trainer('data\Dataset\sleep-cassette', 20, './models/model_mpnn_conv_group_finetuned.pt', './logs/model_mpnn_conv_group_finetuned_log.txt', model_path='./models/model_mpnn_conv_group.pt', device=device, view_strat='group', finetune=True)
    t_loss, t_acc = t.finetune('group')
    t.plot(t_loss, None, t_acc, None, 'group', True)
    
    t = Trainer('data\Dataset\sleep-cassette', 20, './models/model_mpnn_conv_split_finetuned.pt', './logs/model_mpnn_conv_split_finetuned_log.txt', model_path='./models/model_mpnn_conv_split.pt', device=device, view_strat='split', finetune=True)
    t_loss, t_acc = t.finetune('split')
    t.plot(t_loss, None, t_acc, None, 'split', True)