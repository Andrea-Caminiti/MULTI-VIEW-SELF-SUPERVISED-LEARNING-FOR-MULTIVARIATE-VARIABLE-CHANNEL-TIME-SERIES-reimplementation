import torch 
import os
import shutil

if __name__ == '__main__':

    path = 'data/Dataset/You snooze you win'
    j=0
    for i, f in enumerate(os.listdir(path)):
        if not os.path.exists(f'data/Dataset/You snooze you win/Dataset {j}'):
            new_path = f'data/Dataset/You snooze you win/Dataset {j}'
            os.makedirs(new_path)
        shutil.move(os.path.join(path, f), os.path.join(new_path, f))
        if i % 20 == 0 and i!=0:
            j+=1
            
    # path = 'data/Dataset/sleep-cassette'
    # j=0
    # for i, f in enumerate(os.listdir(path)):
    #     if not os.path.exists(f'data/Dataset/sleep-cassette/Dataset {j}'):
    #         new_path = f'data/Dataset/sleep-cassette/Dataset {j}'
    #         os.makedirs(new_path)
    #     shutil.move(os.path.join(path, f), os.path.join(new_path, f))
    #     if i % 20 == 0 and i!=0:
    #         j+=1

        