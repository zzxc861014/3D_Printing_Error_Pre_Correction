import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_generator_3Dpatch import Dataset_3D
from posi_encoding import PositionalEncodingPermute1D

np.random.seed(2022)

def Position_encoding_1D(coord):
    p_enc_per_1d_s = PositionalEncodingPermute1D(10)
    p_enc_per_1d_xy = PositionalEncodingPermute1D(10)

    array = np.zeros((len(coord),10,30))
    array = p_enc_per_1d_s(torch.from_numpy(array))
    array_s = torch.zeros(len(coord),10,1)

    for i, index in enumerate(coord[:,0]):
        array_s[i,:,0] = array[i,:,index]  

    array = np.zeros((len(coord),10,128))
    array = p_enc_per_1d_xy(torch.from_numpy(array))
    array_x = torch.zeros(len(coord),10,1)
    for i, index in enumerate(coord[:,1]):
        array_x[i,:,0] = array[i,:,index]  

    array_y = torch.zeros((len(coord),10,1))
    for i, index in enumerate(coord[:,2]):
        array_y[i,:,0] = array[i,:,index] 

    array = torch.cat((array_s, array_x, array_y), dim=1)
    array = array.unsqueeze(2)
    array = array.unsqueeze(3)

    return array


def main():
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 100
    training_set = Dataset_3D()
    train_loader = DataLoader(training_set, batch_size=batch_size, num_workers=0, drop_last=True)
    correct_per_epo, false_per_epo = [], []
    files = np.linspace(10, 1600, 160)
    # files = ['MAE']
    start_time = time.time()
    for model_name in files:
        model = torch.load('/work/310613060/two_stage/Position_Encoding/checkpoints_PG_128x128/generator_'+str(int(model_name))+'.pth')
        model.to(device)
        model.eval()
        threshold = 0.5
        total_correct, total_false = [], []
        with torch.no_grad():
            for inputs, targets, coord in train_loader:
                coord = Position_encoding_1D(coord)
                x = inputs.cuda().float()
                y = targets.float().numpy()
                coord = coord.cuda().float()
                output = model(x,coord)
                output = output.cpu().numpy()
                output[output>=threshold] = 1
                output[output<threshold] = 0
                x = x.cpu().numpy()
                real_err = x-y
                pred_err = x-output
                correct_rate, false_rate = 0, 0
                for i in range(batch_size):
                    c, f = accuracy_analysis(real_err[i], pred_err[i])
                    correct_rate+=c
                    false_rate+=f
                print(correct_rate/batch_size, false_rate/batch_size)
                total_correct.append(correct_rate/batch_size)
                total_false.append(false_rate/batch_size)
        correct_per_epo.append(sum(total_correct)/len(total_correct)*100)
        false_per_epo.append(sum(total_false)/len(total_false)*100)
        print('correct=', sum(total_correct)/len(total_correct)*100)
        print('false=', sum(total_false)/len(total_false)*100)

    end_time = time.time()
    
    print('runtime: %d min %d sec'%((end_time - start_time)//60,(end_time - start_time)%60))
    plt.plot(files, correct_per_epo, c='green', label='correct%_per_epoch')
    plt.plot(files, false_per_epo, c='red', label='incorrect%_per_epoch')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('percentage')
    plt.savefig('test_accuracyPG_channel_128x128.png')


def accuracy_analysis(real, pred):
    real_err_sum = np.sum(np.abs(real))
    pred_err_sum = np.sum(np.abs(pred))
    pred_correct = np.sum(pred[real==1]==1) + np.sum(pred[real==-1]==-1)
    pred_wrong = np.sum(pred[real==0]==1) + np.sum(pred[real==0]==-1)
    accuracy = (pred_correct/real_err_sum)
    false_rate = (pred_wrong/pred_err_sum)
    return accuracy, false_rate


if __name__ == '__main__':
    main()