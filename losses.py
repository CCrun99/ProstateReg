import numpy as np
from scipy.ndimage import  distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
import torch.nn as nn

class compute_sdf(nn.Module):

    def __init__(self):
        super(compute_sdf, self).__init__()


    def forward(self,img_gt, out_shape):
        """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size,c, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                 -inf|x-y|; x in segmentation
                 +inf|x-y|; x out of segmentation
        normalize sdf to [-1,1]
        """

        img_gt = img_gt.astype(np.uint8)
        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
            for c in range(out_shape[1]):
                posmask = img_gt[b].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

        return torch.tensor(normalized_sdf,requires_grad=True)


class compute_sdf_lesion(nn.Module):

    def __init__(self):
        super(compute_sdf_lesion, self).__init__()

    def forward(self,img_gt, out_shape):
        """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size,c, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                 -inf|x-y|; x in segmentation
                 +inf|x-y|; x out of segmentation
        normalize sdf to [-1,1]
        """

        img_gt = img_gt.astype(np.uint8)
        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
            for c in range(out_shape[1]):
                posmask = img_gt[b].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

        return torch.tensor(normalized_sdf,requires_grad=True)



class SDMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y_pred, y_true):
        smooth = 1e-5

        intersect = torch.sum(y_pred * y_true)

        pd_sum = torch.sum(y_pred ** 2)

        gt_sum = torch.sum(y_true ** 2)

        L_product = 1-(intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)

        L_SDF_AAAI = L_product+ (torch.norm(y_pred - y_true, p=1)) / torch.numel(y_pred)+(torch.norm(y_pred - y_true)) / torch.numel(y_pred)

        loss =  L_SDF_AAAI

        return torch.reshape(loss, (1,))





