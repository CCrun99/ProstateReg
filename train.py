import os
import matplotlib.pyplot as plt
import torch.optim
from monai.transforms import (LoadImaged,EnsureChannelFirstd,ToTensord,Compose,Resized,RandAffined,Orientationd,RandRotated)
from monai.metrics import compute_dice,DiceMetric
from monai.data import Dataset,DataLoader,CacheDataset
from monai.networks.nets import RegUNet
from monai.data.image_reader import Nifti1Image
from monai.networks.nets import  LocalNet
from monai.networks.blocks import Warp,DVF2DDF
from monai.losses import DiceLoss,MultiScaleLoss,BendingEnergyLoss
from utils import getfilename
from losses import compute_sdf,SDMLoss,compute_sdf_lesion

fixed_image_train =  ''     #Training Set Fixed Image Save Path
fixed_label_train =  ''     #Training Set Fixed Label Save Path
moving_image_train = ''     #Training Set Moving Image Save Path
moving_label_train = ''     #Training Set Moving Label Save Path
moving_lesion_train= ''     #Training Set Lesion Image Save Path
fixed_lesion_train = ''     #Training Set Lesion Label Save Path


fixed_image_valid =  ''     #Validation Set Fixed Image Save Path
fixed_label_valid =  ''     #Validation Set Fixed Label Save Path
moving_image_valid = ''     #Validation Set Moving Image Save Path
moving_label_valid = ''     #Validation Set Moving Label Save Path
moving_lesion_valid= ''     #Validation Set Lesion Image Save Path
fixed_lesion_valid = ''     #Validation Set Lesion Label Save Path

filename_train = getfilename([fixed_image_train ,fixed_label_train ,moving_image_train ,moving_label_train,
                              fixed_lesion_train,moving_lesion_train])
filename_valid = getfilename([fixed_image_valid,fixed_label_valid ,moving_image_valid ,moving_label_valid,
                              fixed_lesion_valid,moving_lesion_valid])
keys = ('fixed_image',
            'fixed_label',
            "moving_image",
            "moving_label",
        'fixed_lesion',
        'moving_lesion')
train_trans = Compose([LoadImaged(keys),
                       RandAffined(keys),
                       EnsureChannelFirstd(keys),
                       RandRotated(keys = keys,range_x = [0,0.5],range_y = [0,0.5],range_z = [0,0.5],prob = 0.1),
                       ToTensord(keys),
                       Resized(keys = keys,spatial_size=(128,128,32),mode = 'trilinear ')])
valid_trans = Compose([LoadImaged(keys),
                       EnsureChannelFirstd(keys),
                       ToTensord(keys),
                       Resized(keys = keys,spatial_size=(128,128,32),mode = 'trilinear ')])





train_data = Dataset(filename_train,train_trans)
valid_data = Dataset(filename_valid,valid_trans)
train_loder = DataLoader(dataset = train_data,batch_size=1,shuffle=False)
valid_loder = DataLoader(dataset = valid_data,batch_size=1,shuffle=False)
#


 #build model
model = RegUNet(spatial_dims=3,in_channels=2,num_channel_initial=32,depth=3,
                extract_levels=[0,1,2,3],concat_skip=True).to('cuda')
wary_layer = Warp().to('cuda')
compute_sdf_layer = compute_sdf()
compute_sdf_layer_lesion = compute_sdf_lesion()


#build loss
SDMloss = SDMLoss()
label_loss_fn = DiceLoss()
label_loss_fn = MultiScaleLoss(label_loss_fn,scales = [0,1,2,4,8,16])
regularization_fn = BendingEnergyLoss()
optim = torch.optim.Adam(model.parameters(),1e-3)


def forward(batch_data,model):

    fixed_image = batch_data["fixed_image"].to('cuda')
    fixed_label = batch_data['fixed_label'].to('cuda')
    moving_image = batch_data['moving_image'].to('cuda')
    moving_label = batch_data['moving_label'].to('cuda')
    fixed_lesion = batch_data['fixed_lesion'].to('cuda')
    moving_lesion = batch_data['moving_lesion'].to('cuda')
    ddf = model(torch.cat([moving_image,fixed_image],dim = 1))
    pred_image = wary_layer(moving_image,ddf)
    pre_label = wary_layer(moving_label,ddf)
    lesion_label = wary_layer(moving_lesion,ddf)
    pre_sdm = compute_sdf_layer(pre_label,pre_label.shape).to('cuda')
    true_sdm = compute_sdf_layer(fixed_label,fixed_label.shape).to('cuda')
    #pre_lesion_sdm = compute_sdf_layer_lesion(lesion_label,fixed_lesion.shape).to('cuda')
    #true_lession_sdm = compute_sdf_layer_lesion(fixed_lesion,fixed_lesion.shape).to('cuda')
    #return pred_image,pre_label,pre_sdm,true_sdm,lesion_label,ddf,pre_lesion_sdm,true_lession_sdm
    return pred_image, pre_label, pre_sdm, true_sdm, lesion_label, ddf



# train
dir = ''               #Parameter Save Path
param = 0
max_dice = 0
epochs = 50
max_epoch=0
train_loss = []
valid_loss = []
metric = []


for epoch in range(epochs):
    epoch_total_loss = 0
    step = 0
    epoch_sdm_loss = 0
    epoch_dice_loss = 0
    model.train()
    for batch_data in train_loder:
        step +=1
        optim.zero_grad()

        #pred_image,pred_label,pre_sdm,true_sdm,lesion_label,ddf,pre_lesion_sdm,true_lession_sdm = forward(batch_data,model)
        pred_image, pred_label, pre_sdm, true_sdm, lesion_label, ddf = forward(batch_data, model)
        fixed_image = batch_data['fixed_image'].to('cuda')
        fixed_lesion =batch_data['fixed_lesion'].to('cuda')


        #loss
        sdm_loss = SDMloss(pre_sdm,true_sdm)                          #SDM  LOSS
        #lesion_sdm_loss = SDMloss(pre_lesion_sdm,true_lession_sdm)   #lesion  SDM  LOSS
        fixed_label = batch_data['fixed_label'].to('cuda')
        label_loss = label_loss_fn(pred_label,fixed_label)             #DICE LOSS
        #lesion_loss = label_loss_fn(lesion_label,fixed_lesion)          #lesion  loss
        regularization_loss = regularization_fn(ddf)                 #regularization LOSS
        loss =  sdm_loss+label_loss+regularization_loss               #total  LOSS


        loss.backward()
        optim.step()


        #output loss
        epoch_total_loss = epoch_total_loss +loss
        epoch_sdm_loss = epoch_sdm_loss+sdm_loss
        epoch_dice_loss  = epoch_dice_loss +label_loss


    epoch_mean_loss = ((epoch_total_loss/len(train_data)).cpu()).detach().numpy()
    print(f'The SDM LOSS of the {epoch+1} epoch in the training set is :',(epoch_sdm_loss/len(train_data)).item())
    print(f'The DICE LOSS of the {epoch+1} epoch in the training set is ',epoch_dice_loss/len(train_data))
    print(f"The TOTAL LOSS of the {epoch+1} epoch in the training set is :", epoch_mean_loss)
    train_loss.append(epoch_mean_loss)



    model.eval()
    valid_epoch_loss= 0
    smv = 0
    dmv = 0
    dice = 0
    with torch.no_grad():
        for batch_data1 in valid_loder:
            pred_image,pred_label,pre_sdm,true_sdm,lesion_label,ddf = forward(batch_data1,model)
            #pred_image, pred_label, pre_sdm, true_sdm, lesion_label, ddf,pre_lesion_sdm, true_lession_sdm = forward(batch_data1,model)
            fixed_image = batch_data1['fixed_image'].to('cuda')
            fixed_label = batch_data1['fixed_label'].to('cuda')
            fixed_lesion = batch_data1['fixed_lesion'].to('cuda')


            sdm_loss = SDMloss(pre_sdm,true_sdm)
            #lesion_loss = label_loss_fn(lesion_label, fixed_lesion)
            label_loss = label_loss_fn(pred_label, fixed_label)
            regularization_loss = regularization_fn(ddf)
            loss =   sdm_loss+label_loss+regularization_loss


            valid_epoch_loss = valid_epoch_loss+loss
            dice+=compute_dice(y_pred = pred_label,y = fixed_label)  #metric

        valid_mean_loss =((valid_epoch_loss/len(valid_data)).cpu()).detach().numpy()
        dice = (dice / len(valid_data))[0][0].cpu().detach().numpy()
        metric.append(dice)
        print(f'The DICE of the {epoch+1} epoch in the validation set is:', dice)
        print(f"The TOTAL LOSS of the {epoch+1} epoch in the validation set is:", valid_mean_loss)
        valid_loss.append(valid_mean_loss)


    if dice>max_dice:
        max_dice = dice
        max_epoch=epoch+1
        param = model.state_dict()
        torch.save(param, dir+f'SMD_{epoch+1}.pth')
    print(f"The best performance is {max_epoch} epoch")








