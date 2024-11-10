import numpy as np
import os
from utils import config
import cupy
from module import EART

def ProjEARTRecon(path,savePath):
    files = os.listdir(path)
    rp, lsp, hsp = config.ScanParam()
    rp['nSize'] = rp['nSize']*2
    rp['pixelSize'] = rp['pixelSize']/2.0
    args = config.getArgs()
    sl, sh, muH2O, muAl, muBone = config.load_spectrum_and_mu(args.low_spectrum, args.high_spectrum)
    os.makedirs(savePath,exist_ok=True)
    for file in files:
        pL = -np.log(np.fromfile(path+'/'+file,dtype=np.float32).reshape(1,lsp['views'],lsp['bins']))
        pH = -np.log(np.fromfile(path+'/../hKvProj/'+file,dtype=np.float32).reshape(1,hsp['views'],hsp['bins']))
        proj = cupy.asarray(np.concatenate((pL,pH),axis=0))
        x0 = cupy.zeros((2,rp['nSize'],rp['nSize']),dtype=cupy.float32)
        x1 = EART.EART2S2M(x0,proj,rp,lsp,hsp,muH2O,muAl,sl,sh,2)
        cupy.asnumpy(x1).tofile(savePath+'/'+file)

def trans2singleRaw(in_path,save_path):
    data = np.load(in_path)
    os.makedirs(save_path,exist_ok=True)
    for i in range(data.shape[0]):
        data[i,...].tofile(save_path+'/{:03d}.raw'.format(i))

def train():
    args = config.getArgs()
    print('--------------------------train data----------------------------------')
    trans2singleRaw(args.npysrc_data+'/train/highkVpTransmission.npy',
                    args.data_path+'/train/hKvproj')
    trans2singleRaw(args.npysrc_data+'/train/lowkVpTransmission.npy',
                    args.data_path+'/train/lKvproj')
    trans2singleRaw(args.npysrc_data+'/train/highkVpImages.npy',
                    args.data_path+'/train/hKvImage')
    trans2singleRaw(args.npysrc_data+'/train/lowkVpImages.npy',
                    args.data_path+'/train/lKvImage')
    trans2singleRaw(args.npysrc_data+'/train/Phantom_Adipose.npy',
                    args.data_path+'/train/A')
    trans2singleRaw(args.npysrc_data+'/train/Phantom_Calcification.npy',
                    args.data_path+'/train/C')
    trans2singleRaw(args.npysrc_data+'/train/Phantom_Fibroglandular.npy',
                    args.data_path+'/train/F')
    ProjEARTRecon(args.data_path+'/train/lKvProj',args.data_path+'/train/first_eart2m2')
    print('------------------------train finish----------------------------------')


def val():
    args = config.getArgs()
    print('------------------------validation data----------------------------------')
    trans2singleRaw(args.npysrc_data+'/val/highkVpTransmission.npy',
                    args.data_path+'/val/hKvproj')
    trans2singleRaw(args.npysrc_data+'/val/lowkVpTransmission.npy',
                    args.data_path+'/val/lKvproj')
    trans2singleRaw(args.npysrc_data+'/val/highkVpImages.npy',
                    args.data_path+'/val/hKvImage')
    trans2singleRaw(args.npysrc_data+'/val/lowkVpImages.npy',
                    args.data_path+'/val/lKvImage')
    ProjEARTRecon(args.data_path+'/val/lKvproj',args.data_path+'/val/first_eart2m2')
    print('-----------------------validation finish----------------------------------')

def start_kit():
    args = config.getArgs()
    print('------------------------start_kit data----------------------------------')
    trans2singleRaw(args.npysrc_data+'/starting_kit/highkVpTransmission.npy',
                    args.data_path+'/starting_kit/hKvproj')
    trans2singleRaw(args.npysrc_data+'/starting_kit/lowkVpTransmission.npy',
                    args.data_path+'/starting_kit/lKvproj')
    trans2singleRaw(args.npysrc_data+'/starting_kit/Phantom_Adipose.npy',
                    args.data_path+'/starting_kit/A')
    trans2singleRaw(args.npysrc_data+'/starting_kit/Phantom_Calcification.npy',
                    args.data_path+'/starting_kit/C')
    trans2singleRaw(args.npysrc_data+'/starting_kit/Phantom_Fibroglandular.npy',
                    args.data_path+'/starting_kit/F')
    trans2singleRaw(args.npysrc_data+'/starting_kit/highkVpImages.npy',
                    args.data_path+'/starting_kit/hKvImage')
    trans2singleRaw(args.npysrc_data+'/starting_kit/lowkVpImages.npy',
                    args.data_path+'/starting_kit/lKvImage')
    ProjEARTRecon(args.data_path+'/starting_kit/lKvproj',args.data_path+'/starting_kit/first_eart2m2')
    print('------------------------start_kit finish----------------------------------')

def test():
    args = config.getArgs()
    print('--------------------------test data----------------------------------')
    trans2singleRaw(args.npysrc_data+'/test/highkVpTransmission.npy',
                    args.data_path+'/test/hKvproj')
    trans2singleRaw(args.npysrc_data+'/test/lowkVpTransmission.npy',
                    args.data_path+'/test/lKvproj')
    ProjEARTRecon(args.data_path+'/test/lKvproj',args.data_path+'/test/first_eart2m2')
    print('--------------------------test finish---------------------------------')

if __name__ == '__main__':
    train()
    val()
    start_kit()
    # test()