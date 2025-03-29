from model.model import Model
from dataset.datatset import createDataset
from losses import TS2VecLoss
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from sklearn.decomposition import PCA

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
        size = (1,) if 'sleep-cassette' in datapath else (2,)
        self.eval_idxs = np.random.randint(len(os.listdir(datapath)), size=size)
        self.eval_folders = [os.listdir(self.datapath)[i] for i in self.eval_idxs]
        self.train_folders = [folder for folder in os.listdir(self.datapath) if folder not in self.eval_folders]
        self.epochs = epochs
        self.save_path=save_path
        self.log_path = log_path
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-5, weight_decay=1e-4)
        if finetune:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = TS2VecLoss(0.01)
        self.device = device

    def pretrain(self, view_strat): #To be used with You Snooze You Win dataset
        epoch_loss = []
        eval_loss = []
        with open(self.log_path, 'w') as log:
            log.write('#'*25 + ' TRAINING REPORT You Snooze You Win Dataset ' + '#'*25 + '\n')
            log.write(f'Training with MPNN_GAT: {self.attention}\n')
            log.write(f'Training with view strategy: {view_strat}\n')
            
            for epoch in tqdm(range(self.epochs), desc='Training...'):
                self.model = self.model.train()
                datas_loss=[]
                for dataset in tqdm(self.train_folders, desc=f'Epoch {epoch}', leave=False):
                    dset = createDataset(os.path.join(self.datapath, dataset))
                    dloader = torch.utils.data.DataLoader(dset, batch_size = 1, collate_fn=collate)
                    ls = [] # batch loss
                    for batch in tqdm(dloader, desc='Dataloader...', leave=False):
                        samples, labels = batch
                        idxs = np.concatenate([np.where(labels == 0)[0][:40], np.where(labels == 1)[0][:40], np.where(labels == 2)[0][:40], np.where(labels == 3)[0][:40], np.where(labels == 4)[0][:40]])
                        idxs = np.random.choice(idxs, len(idxs))
                        samples = samples.permute(1, 0, 2)[idxs]
                        labels = labels[idxs].long()  
                        self.optimizer.zero_grad()
                        b, ch, t = samples.shape
                        
                        if view_strat == 'split':
                            loss = 0.0                   
                            views = [samples[:, c, :].unsqueeze(1) for c in range(ch)]         
                            embeddings = torch.stack([self.model.encode(v.to(self.device)) for v in views])
                            for i in range(ch):
                                for j in range(i+1, ch): 
                                    loss += self.loss(embeddings[i], embeddings[j])
                        
                        else:
                            permuted_indices = torch.randperm(ch)
                            sep = torch.randint(low=2, high=ch-1, size=(1,))
                            group1 = permuted_indices[:sep]
                            group2 = permuted_indices[sep:]
                            v1, v2 = samples[:, group1, :], samples[:, group2, :]
                            v1 = v1.view(b*len(group1), 1, t)
                            v2 = v2.view(b*len(group2), 1, t) 
                            
                            embedding1 = self.model.encode(v1.to(self.device))
                            embedding2 = self.model.encode(v2.to(self.device))
                            
                            embedding1 = self.model.message(embedding1.permute(2, 0, 1), b, len(group1)).permute(1, 0, 2)
                            embedding2 = self.model.message(embedding2.permute(2, 0, 1), b, len(group2)).permute(1, 0, 2)
                            
                            embedding1 = embedding1.view(b, len(group1), 31, 128).sum(dim=1)
                            embedding2 = embedding2.view(b, len(group2), 31, 128).sum(dim=1)
                            
                            embedding1 = self.model.mpnn.rout(embedding1)
                            embedding2 = self.model.mpnn.rout(embedding2)
                            
                            loss = self.loss(embedding1.permute(0, 2, 1), embedding2.permute(0, 2, 1))                 
                        
                        loss.backward()
                        for name, param in self.model.named_parameters():
                             if param.grad is not None:
                                 print(f"{name} | Grad Mean: {param.grad.abs().mean().item():.6f} | Grad Std: {param.grad.std().item():.6f}")
                        
                        self.optimizer.step()
                        ls.append(loss.item())
                        print(ls[-1])
                    datas_loss.append(np.mean(ls).item())
                epoch_loss.append(np.mean(datas_loss).item())
                log.write(f'Epoch {epoch}: \n\tTraining Loss: {epoch_loss[-1]}')
                self.model = self.model.eval()
                e_loss, *_ = self.eval(self.eval_folders, epoch, view_strat)
                
                eval_loss.append(e_loss)
               
                log.write(f'\n\tValid Loss: {e_loss}\n')
               
        
        torch.save(self.model.state_dict(), self.save_path)

        return epoch_loss, eval_loss 

    def eval(self, eval_folders, i, view_strat, finetune=False):
        datas_loss=[]
        datas_acc = []
        datas_prec = []
        datas_rec = []
        datas_f = []
        for dataset in tqdm(eval_folders, desc=f'Evaluation after epoch {i}...', leave=False):
            with torch.no_grad():
                dset = createDataset(os.path.join(self.datapath, dataset), aug = None)
                dloader = torch.utils.data.DataLoader(dset, batch_size = 1, collate_fn=collate)
                ls = [] # batch loss
                preds, gt = [], []
                
                for batch in tqdm(dloader, desc='Dataloader...', leave=False):
                    samples, labels = batch
                    idxs = np.concatenate([np.where(labels == 0)[0][:40], np.where(labels == 1)[0][:40], np.where(labels == 2)[0][:40], np.where(labels == 3)[0][:40], np.where(labels == 4)[0][:40]])
                    idxs = np.random.choice(idxs, len(idxs))
                    samples = samples.permute(1, 0, 2)[idxs]
                    labels = labels[idxs].long().to(self.device)  
                    b, ch, t = samples.shape
                    
                    if view_strat == 'split':
                        loss = 0.0                   
                        views = [samples[:, c, :].unsqueeze(1) for c in range(ch)]         
                        embeddings = torch.stack([self.model.encode(v.to(self.device)) for v in views])
                        if finetune:
                            res = self.model.classify(embeddings.permute(1, 2, 3, 0))
                            loss = self.loss(res, labels)
                        else:
                            for i in range(ch):
                                for j in range(i+1, ch): 
                                    loss += self.loss(embeddings[i], embeddings[j])
                    
                    else:
                        if finetune:
                            group1 = [0]
                            group2 = [1]
                            
                        else:
                            permuted_indices = torch.randperm(ch)
                            sep = torch.randint(low=2, high=ch-1, size=(1,))
                            group1 = permuted_indices[:sep]
                            group2 = permuted_indices[sep:]
                        v1, v2 = samples[:, group1, :], samples[:, group2, :]
                        
                        v1 = v1.view(b*len(group1), 1, t)
                        v2 = v2.view(b*len(group2), 1, t) 
                    
                        embedding1 = self.model.encode(v1.to(self.device))
                        embedding2 = self.model.encode(v2.to(self.device))
                        
                        embedding1 = self.model.message(embedding1.permute(2, 0, 1), b, len(group1)).permute(1, 0, 2)
                        embedding2 = self.model.message(embedding2.permute(2, 0, 1), b, len(group2)).permute(1, 0, 2)
                        
                        embedding1 = embedding1.view(b, len(group1), 31, 128).sum(dim=1)
                        embedding2 = embedding2.view(b, len(group2), 31, 128).sum(dim=1)
                        
                        embedding1 = self.model.mpnn.rout(embedding1).permute(0, 2, 1)
                        embedding2 = self.model.mpnn.rout(embedding2).permute(0, 2, 1)
                        if finetune:
                            embeddings = torch.stack((embedding1, embedding2)).permute(1, 2, 3, 0)
                            res = self.model.classify(embeddings)
                            loss = self.loss(res, labels)
                        else:
                            loss = self.loss(embedding1, embedding2)               
        
                    ls.append(loss.item())
                    if finetune:
                        preds.append(res.argmax(dim=1).cpu())
                        gt.append(labels.cpu())

                datas_loss.append(np.mean(ls).item())
                if finetune:
                    preds = np.concatenate(preds, axis=None)
                    gt = np.concatenate(gt, axis=None)
                    datas_acc.append(balanced_accuracy_score(gt, preds))
                    prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='weighted', labels=np.arange(5, step=1))
                    datas_prec.append(prec)
                    datas_rec.append(rec)
                    datas_f.append(f)
                else:
                    datas_acc = []
                    datas_prec = []
                    datas_rec = []
                    datas_f = []
        return np.mean(datas_loss).item(), np.mean(datas_acc).item(), np.mean(datas_prec).item(), np.mean(datas_rec).item(), np.mean(datas_f).item()
      
    def plot(self, train_loss, valid_loss, train_acc, view_strat, finetune):
        
        if train_acc:
            fig, (loss, acc) = plt.subplots(2, 1, sharex=True) 
            fig.set_figwidth(15)
            fig.set_figheight(5)

            loss.set_title('Losses' if valid_loss else 'Training Loss')
            loss.plot(train_loss, label='Training Loss')
            if valid_loss:
                loss.plot(valid_loss, label='Evaluation Loss')
                loss.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            acc.set_title('Training accuracy')
            acc.plot(train_acc, label='Training Accuracy')

        else: 
            fig = plt.figure()
            fig.set_figwidth(15)
            fig.set_figheight(5)
            plt.title('Losses' if valid_loss else 'Training Loss')
            plt.plot(train_loss, label='Training Loss')
            if valid_loss:
                plt.plot(valid_loss, label='Evaluation Loss')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

       
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3, weight_decay=1e-2)
        self.loss = torch.nn.CrossEntropyLoss()
        epoch_loss = []
        train_acc = []
        with open(self.log_path, 'w') as log:
            log.write('#'*25 + ' FINETUNING REPORT Sleep-Cassette Dataset ' + '#'*25 + '\n')
            log.write(f'Training with MPNN_GAT: {self.attention}\n')
            log.write(f'Training with view strategy: {view_strat}\n')
            for epoch in tqdm(range(self.epochs), desc='Training...'):
                self.model = self.model.train()
                datas_loss=[]
                datas_acc = []
                for dataset in tqdm(self.train_folders, desc=f'Epoch {epoch}', leave=False):
                    dset = createDataset(os.path.join(self.datapath, dataset))
                    dloader = torch.utils.data.DataLoader(dset, batch_size = 1, collate_fn=collate)
                    ls = [] # batch loss
                    preds, gt = [], []
                    
                    for batch in tqdm(dloader, desc='Dataloader...', leave=False):
                        samples, labels = batch
                        idxs = np.concatenate([np.where(labels == 0)[0][:40], np.where(labels == 1)[0][:40], np.where(labels == 2)[0][:40], np.where(labels == 3)[0][:40], np.where(labels == 4)[0][:40]])
                        idxs = np.random.choice(idxs, len(idxs))
                        samples = samples.permute(1, 0, 2)[idxs]
                        labels = labels[idxs].long()  
                        self.optimizer.zero_grad()
                        b, ch, t = samples.shape
                        
                        if view_strat == 'split':
                            loss = 0.0                   
                            views = [samples[:, c, :].unsqueeze(1) for c in range(ch)]         
                            embeddings = torch.stack([self.model.encode(v.to(self.device)) for v in views])
                            res = self.model.classify(embeddings.permute(1, 2, 3, 0))
                            loss = self.loss(res, labels)
                        
                        else:
                            
                            v1, v2 = samples[:, 0, :].unsqueeze(1), samples[:, 1, :].unsqueeze(1)
                            embedding1 = self.model.encode(v1.to(self.device))
                            embedding2 = self.model.encode(v2.to(self.device))
                            
                            embedding1 = self.model.message(embedding1.permute(2, 0, 1), b, 1).permute(1, 0, 2)
                            embedding2 = self.model.message(embedding2.permute(2, 0, 1), b, 1).permute(1, 0, 2)
                            
                            embedding1 = embedding1.view(b, 1, 31, 128).sum(dim=1)
                            embedding2 = embedding2.view(b, 1, 31, 128).sum(dim=1)
                            
                            embedding1 = self.model.mpnn.rout(embedding1).permute(0, 2, 1)
                            embedding2 = self.model.mpnn.rout(embedding2).permute(0, 2, 1)
                            
                            embeddings = torch.stack((embedding1, embedding2)).permute(1, 2, 3, 0)
                            res = self.model.classify(embeddings)
                            loss = self.loss(res, labels.to(self.device))
                            
                        loss.backward()
                        for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    print(f"{name} | Grad Mean: {param.grad.abs().mean().item():.6f} | Grad Std: {param.grad.std().item():.6f}")
                        
                        self.optimizer.step()
            
                        ls.append(loss.item())
                        preds.append(res.argmax(dim=1).cpu())
                        gt.append(labels.cpu())
             
                    preds = np.concatenate(preds, axis=None)
                    gt = np.concatenate(gt, axis=None)
                    datas_loss.append(np.mean(ls).item())
                    datas_acc.append(balanced_accuracy_score(gt, preds))
                epoch_loss.append(np.mean(datas_loss).item())
                train_acc.append(np.mean(datas_acc).item())
                
                log.write(f'Epoch {epoch}: \n\tTraining Loss: {epoch_loss[-1]}\n\tTraining Accuracy: {train_acc[-1]}\n')
            self.model = self.model.eval()
            e_loss, e_acc, prec, rec, f = self.eval(self.eval_folders, epoch, view_strat, True)
            
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
        
    t=Trainer('data\Dataset\You snooze you win', 10, './models/model_mpnn_conv_group.pt', './logs/model_mpnn_conv_group.log.txt', Model(6, view_strat='group'), device=device, view_strat='group')
    t_loss, val_loss = t.pretrain('group')
    t.plot(t_loss, val_loss, None, 'group', False)
    
    t = Trainer('data\Dataset\sleep-cassette', 10, './models/model_mpnn_conv_group_finetuned.pt', './logs/model_mpnn_conv_group_finetuned_log.txt', model_path='./models/model_mpnn_conv_group.pt', device=device , view_strat='group', finetune=True)
    t_loss, t_acc = t.finetune('group')
    t.plot(t_loss, None, t_acc, 'group', True)
    
    t=Trainer('data\Dataset\You snooze you win', 10, 'model_split.pt', './logs/model_split.log.txt', Model(6, view_strat='split'), device=device, view_strat='split')
    t_loss, val_loss = t.pretrain('split')
    t.plot(t_loss, val_loss, None, 'split', False)
    
    t = Trainer('data\Dataset\sleep-cassette', 10, './models/model_split_finetuned.pt', './logs/model_split_finetuned_log.txt', model_path='./models/model_split.pt', device=device, view_strat='split', finetune=True)
    t_loss, t_acc = t.finetune('split')
    t.plot(t_loss, None, t_acc, None, 'split', True)