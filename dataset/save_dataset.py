from datatset import createDataset
import torch 
import os
import shutil

if __name__ == '__main__':
    
    path = 'data/Dataset/You snooze you win/training_fif'
    j=0
    for i, f in enumerate(os.listdir(path)):
        if not os.path.exists(f'data/Dataset/You snooze you win/Dataset {j}'):
            new_path = f'data/Dataset/You snooze you win/Dataset {j}'
            os.makedirs(new_path)
        shutil.move(os.path.join(path, f), os.path.join(new_path, f))
        if i % 50 == 0 and i!=0:
            j+=1
            
    path = 'data/Dataset/sleep-cassette'
    j=0
    for i, f in enumerate(os.listdir(path)):
        if not os.path.exists(f'data/Dataset/sleep-cassette/Dataset {j}'):
            new_path = f'data/Dataset/sleep-cassetteDataset {j}'
            os.makedirs(new_path)
        shutil.move(os.path.join(path, f), os.path.join(new_path, f))
        if i % 50 == 0 and i!=0:
            j+=1
            
    # PS18 aka You Snooze You Win
    for i in range(13):
        d_set = createDataset(f'data/Dataset/You snooze you win/Dataset {i+1}')
        torch.save(d_set, f'Dataset {i+1}.pt')
        print('\n'*5 + '#'*25 + '\n' + f'SAVED DATASET {i+1}' + '\n' + '#'*25 + '\n'*5  )
        del d_set

    
    # SC
    for i in range(3):
        d_set = createDataset(f'data/Dataset/sleep-cassette/Dataset {i+1}')
        torch.save(d_set, f'Dataset {i+1}.pt')
        print('\n'*5 + '#'*25 + '\n' + f'SAVED DATASET {i+1}' + '\n' + '#'*25 + '\n'*5  )
        del d_set
        