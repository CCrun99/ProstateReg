import os
from utils import getfilename
import matplotlib.pyplot as plt
import torch.optim
from monai.transforms import (LoadImaged,AddChanneld,ToTensord,Compose,Resized)
from monai.data import Dataset,DataLoader,CacheDataset,write_nifti
from monai.data.image_reader import Nifti1Image
from monai.networks.nets import  LocalNet
from monai.networks.blocks import Warp,DVF2DDF
from mindspore.nn.metrics import HausdorffDistance
from mindspore.nn.metrics import MeanSurfaceDistance
from monai.losses import DiceLoss,MultiScaleLoss,BendingEnergyLoss
from torch.nn import  MSELoss
from utils import getfilename
from losses import compute_sdf,SDMLoss
from monai.metrics import compute_dice,compute_hausdorff_distance,compute_average_surface_distance
from monai.networks.nets import RegUNet
import numpy as np

fixed_image_test =  ''       #Test Set Fixed Image Save Path
fixed_label_test =  ''       #Test Set Fixed Label Save Path
moving_image_test= ''        #Test Set Moving Image Save Path
moving_label_test = ''       #Test Set Moving Label Save Path
moving_lesion_test= ''       #Test Set Lesion Image Save Path
fixed_lesion_test = ''       #Test Set Lesion Label Save Path

filename_test = getfilename([fixed_image_test ,fixed_label_test ,moving_image_test ,moving_label_test,
                             fixed_lesion_test,moving_lesion_test])

keys = ('fixed_image',
            'fixed_label',
            "moving_image",
            "moving_label",
            'fixed_lesion',
            'moving_lesion')

test_trans = Compose([LoadImaged(keys),AddChanneld(keys),ToTensord(keys),Resized(keys = keys,spatial_size=(128,128,32),mode = 'trilinear ')])

test_data = Dataset(filename_test,test_trans)
test_loder = DataLoader(dataset = test_data,batch_size=1,shuffle=False)



model = RegUNet(spatial_dims=3,in_channels=2,num_channel_initial=32,depth=3,
                extract_levels=[0,1,2,3],concat_skip=True).to('cuda')
wary_layer = Warp().to('cuda')
x= 0
path = ''         #Model Parameter Path
save_path = ''     #Image Save Directory
model.load_state_dict(torch.load(path))
model.eval()
metric = []
lesion_dice_metric = []

HD=[]
HD_lesion = []
MSD=[]
MSD_lesion = []
i = 0
for batch_data in test_loder:
    i=i+1
    fixed_image = batch_data["fixed_image"].to('cuda')
    fixed_label = batch_data['fixed_label'].to('cuda')
    moving_image = batch_data['moving_image'].to('cuda')
    moving_label = batch_data['moving_label'].to('cuda')
    fixed_lesion = batch_data['fixed_lesion'].to('cuda')
    moving_lesion = batch_data['moving_lesion'].to('cuda')
    ddf = model(torch.cat([moving_image, fixed_image], dim=1))

    #Predictive image
    pred_image = wary_layer(moving_image,ddf)
    pre_label = wary_layer(moving_label,ddf)
    pre_lesion = wary_layer(moving_lesion,ddf)

    #Calculate dice
    dice = compute_dice(y_pred=pre_label, y=fixed_label).item()
    lesion_dice = compute_dice(y_pred=pre_lesion, y=fixed_lesion).item()
    ddf = ddf.cpu()

    #Calculate HD
    hd = HausdorffDistance()
    hd.clear()
    hd.update(pre_label.cpu().detach().numpy().squeeze(), fixed_label.cpu().detach().numpy().squeeze(), 0)
    distance = hd.eval()
    HD.append(distance)
    hd_lesion = HausdorffDistance()
    hd_lesion.clear()
    hd_lesion.update(pre_lesion.cpu().detach().numpy().squeeze(), fixed_lesion.cpu().detach().numpy().squeeze(), 0)
    distance_lesion = hd_lesion.eval()
    HD_lesion.append(distance_lesion)

    #Calculate MSD
    metric_MSD_lesion = MeanSurfaceDistance()
    metric_MSD_lesion.clear()
    metric_MSD_lesion.update(pre_lesion.cpu().detach().numpy().squeeze(), fixed_lesion.cpu().detach().numpy().squeeze(), 0)
    distance_MSD_lesion = metric_MSD_lesion.eval()
    MSD_lesion.append(distance_MSD_lesion)
    metric_MSD = MeanSurfaceDistance()
    metric_MSD.clear()
    metric_MSD.update(pre_label.cpu().detach().numpy().squeeze(), fixed_label.cpu().detach().numpy().squeeze(),0)
    distance_MSD = metric_MSD.eval()
    MSD.append(distance_MSD)
    write_nifti(pred_image,path+f'pre_moving_image{i}.nii.gz')
    write_nifti(pre_label, path + f'pre_moving_label{i}.nii.gz')

print("dice  mean: ",np.mean(metric))
print("dice  std: ",np.std(metric))
print("lesion dice  mean: ", np.mean(lesion_dice_metric))
print("lesion dice  std: ", np.std(lesion_dice_metric))
print("HD  mean: ",np.mean(HD))
print("HD  std: ",np.std(HD))
print("HD_lesion  mean: ", np.mean(HD_lesion))
print("HD_lesion  std: ", np.std(HD_lesion))
print("MSD  mean: ",np.mean(MSD))
print("MSD  std: ",np.std(MSD))
print("MSD_lesion  mean: ", np.mean(MSD_lesion))
print("MSD_lesion  std: ", np.std(MSD_lesion))
print('****************************************')
print('****************************************')