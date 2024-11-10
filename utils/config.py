import numpy as np
import cupy
import argparse

def ScanParam():
    # recon pramaters
    rp = {'nSize':512,
          'pixelSize':18.0/512.0
    }
    # low energy scan paramaters
    lsp = {'sod':50.0,
           'odd':50.0,
           'start':0.5,
           'views':256,
           'bins':1024,
           'numAngle':256}
    # high energy scan paramaters
    hsp = {'sod':50.0,
           'odd':50.0,
           'start':0.0,
           'views':256,
           'bins':1024,
           'numAngle':256}
    lsp['sdd'] = lsp['odd']+lsp['sod']
    hsp['sdd'] = hsp['odd']+hsp['sod']
    lsp['cellsize'] = 0.03574#(2*np.tan(np.arcsin((rp['pixelSize']/2.0*rp['nSize'])/lsp['sod']))*lsp['sdd'])/lsp['bins']
    # lsp['cellsize'] = lsp['cellsize'].astype(float)
    hsp['cellsize'] = 0.03574#(2*np.tan(np.arcsin((rp['pixelSize']/2.0*rp['nSize'])/hsp['sod']))*lsp['sdd'])/lsp['bins']
    # hsp['cellsize'] = hsp['cellsize'].astype(float)
    return rp,lsp,hsp

def load_spectrum_and_mu(lpath,hpath):

    lowKv = np.load(lpath)
    highKv = np.load(hpath)
    sl = cupy.asarray(lowKv[1])
    # muH2O_low = lowKv[2]
    # muAl_low = lowKv[3]
    # muBone_low = lowKv[4]
    sh = cupy.asarray(highKv[1])
    muH2O = cupy.asarray(highKv[2])
    muAl = cupy.asarray(highKv[3])
    muBone = cupy.asarray(highKv[4])
    return sl,sh,muH2O,muAl,muBone

def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path',type=str,default=r'D:\BaiduNetdiskDownload\AAPM\label')
    args.add_argument('--npysrc_data',type=str,default=r'D:\BaiduNetdiskDownload\AAPM\npy')
    args.add_argument('--low_spectrum',type=str,default=r'D:\BaiduNetdiskDownload\AAPM\npy\starting_kit/model_data_50kVp.npy')
    args.add_argument('--high_spectrum',type=str,default=r'D:\BaiduNetdiskDownload\AAPM\npy\starting_kit/model_data_80kVp.npy')
    return args.parse_args()
