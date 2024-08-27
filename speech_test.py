

#@title Importation and Set Deterministic Algorithms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import torch.nn.functional as F
from monai.utils import MetricReduction
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from lib.Pre_trained_vit_with_new_ref import PRORED
from lib.vit_config import config_vit


torch.use_deterministic_algorithms(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(model, test, display):
        
        #@title Running Parameters
        
        k_folds = 5
        no_classes = 7
        
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        hausdorff_metric = HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=None, directed=False, reduction=MetricReduction.MEAN, get_not_nans=False)

        
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.use_deterministic_algorithms(False)
    
        torch.use_deterministic_algorithms(False)
        
        model=model.cuda()
        'Testing the test set in the way in which the evaluation set is tested'
        
        
        metric_values = list()

        tes=['aa','ah','br','gc','mr']
        df=pd.DataFrame(columns=['Volunteer_test','Frame','Class','Dice','HD','Model'])
        for i in range(k_folds):
            test_vol=tes[i]
            if test:
                model.load_state_dict(torch.load(f'./model_pth/best_metric_model_{i}.pth', map_location=torch.device('cpu')))  
                model.to(device)
           
            model.eval()
            test_set = torch.load(f"./test_dataset/test_{i+1}_1.pt", map_location=torch.device('cpu'))
            testloaderCV = torch.utils.data.DataLoader(test_set, shuffle=False)
           
            outputs = np.zeros([1,7,256,256])
            k=0

            for test_data in testloaderCV:
                k+=1
                test_images, test_labels = test_data['img'].to(device), test_data['seg'].to(device)
        
                test_outputs  = model(test_images)
        
                test_sum = 0
                for sub in test_outputs:
                    test_sum+=sub
                test_outputs = torch.argmax(test_sum, dim=1)
                test_outputs = F.one_hot(test_outputs, num_classes = -1)
                test_outputs = torch.permute(test_outputs, (0, 3, 1, 2))
                
         
                test_labels = F.one_hot(test_labels, num_classes = no_classes)
                test_labels = torch.permute(test_labels, (0, 1, 4, 2, 3))
                test_labels = torch.squeeze(test_labels, dim=1)
        
                dice_metric(y_pred=test_outputs, y=test_labels)
                pp=dice_metric(y_pred=test_outputs, y=test_labels)
                m=torch.mean(pp,0,True)
                
                
                pp_HD=hausdorff_metric(y_pred=test_outputs, y=test_labels)
                
                if test_data == 0:
                    outputs = test_outputs.detach().cpu().numpy()
                    print(str(np.shape(outputs)) + 'outputs 1')
                else:
                    outputs = np.append(outputs, test_outputs.detach().cpu().numpy(), axis = 0)
                
                if display:
                    plt.subplots(1,2,figsize=(8,8))
                    plt.subplot(1,2,1)
                    plt.title(f'Ground Truth-{k}')
                    plt.imshow(np.squeeze(np.argmax(np.squeeze(test_labels[0].cpu()), axis=0)))
                    plt.subplot(1,2,2)
                    plt.title('Prediction'+str(m))
                    plt.imshow(np.squeeze(np.argmax(np.squeeze(test_outputs[0].cpu()), axis=0)))
                    plt.show()
                    
                            
                result=[
                        [test_vol,k,'Head',float(m[0,1]),float(pp_HD[0,1]),'RPORED'],
                        [test_vol,k,'Soft-palate',float(m[0,2]),float(pp_HD[0,2]),'RPORED'],
                        [test_vol,k,'Jaw',float(m[0,3]),float(pp_HD[0,3]),'RPORED'],
                        [test_vol,k,'Tongue',float(m[0,4]),float(pp_HD[0,4]),'RPORED'],
                        [test_vol,k,'Vocal-Tract',float(m[0,5]),float(pp_HD[0,5]),'RPORED'],
                        [test_vol,k,'Tooth-space',float(m[0,6]),float(pp_HD[0,6]),'RPORED']
                        ]
                
                df1=pd.DataFrame(result,columns=df.columns)
                df=df.append(df1)
                    
        
        
            print(str(np.shape(outputs)) + 'outputs at end')
        
            metric = dice_metric.aggregate().item()
            metric1 = dice_metric
            print(metric)
            #sys.exit()
            path="./pred"
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                print(f"The new directory {path} is created!")
            np.save(path  + f'/Sub_{i+1}_outputs', outputs)
            dice_metric.reset()
            print(type(metric1))
            metric_values.append(metric)
            
            
        del model
        df.to_csv('./pred/test_PRORED_.csv',index=True)
        print('Dice',df['Dice'].mean(),'HD',df['HD'].mean())

if __name__ == '__main__':
    net = PRORED(config_vit, img_size=256, num_classes=7).cuda()  
    inference(model=net, test=True, display=True)