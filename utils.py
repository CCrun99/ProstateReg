import numpy as np
import os
def image_path(path):

    for root,dirs,files in os.walk(path):
        pass
    return files

def getfilepath(path_list):

    return [ sorted(image_path(path)) for path in path_list]

def getfilename(path_list):
    filename = []
    path = getfilepath(path_list)
    for fixed_image,fixed_label,moving_image,moving_label,fixed_lesion,moving_lesion in zip(path[0],path[1],path[2],path[3],path[4],path[5]):

        filename.append({
            'fixed_image':os.path.join(path_list[0],fixed_image),
            'fixed_label':os.path.join(path_list[1],fixed_label),
            "moving_image":os.path.join(path_list[2],moving_image),
            "moving_label":os.path.join(path_list[3],moving_label),
            "fixed_lesion": os.path.join(path_list[4], fixed_lesion),
            "moving_lesion": os.path.join(path_list[5], moving_lesion)
        })
    return filename



def Get_Ja(displacement):
    '''
    '''

    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    D = D1 - D2 + D3

    return D