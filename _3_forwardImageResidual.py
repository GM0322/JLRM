import numpy as np
import os
import torch
from module import operator
import matplotlib.pyplot as plt

def residualRecon(path,train=True):
    ct = operator.CT()
    files = os.listdir(path)
    os.makedirs(path+'/../err_0',exist_ok=True)
    if train:
        os.makedirs(path+'/../err_gt',exist_ok=True)
    for i,file in enumerate(files):
        A = torch.from_numpy(np.fromfile(path+'/../Model_0/'+file,
                            dtype=np.float32)).view(1,1,ct.rp['nSize'],ct.rp['nSize'])
        C = torch.from_numpy(np.fromfile(path+'/../Model_1/'+file,
                            dtype=np.float32)).view(1,1,ct.rp['nSize'],ct.rp['nSize'])
        F = torch.from_numpy(np.fromfile(path+'/../Model_2/'+file,
                            dtype=np.float32)).view(1,1,ct.rp['nSize'],ct.rp['nSize'])
        NNrecon = torch.cat([A,C,F],dim=1).cuda()
        NNproj = ct.spectrum_proj_forward(NNrecon)
        pL = torch.from_numpy(np.fromfile(path+'/../lKvproj/'+file,
                            dtype=np.float32)).view(1,1,ct.lsp['views'],ct.lsp['bins'])
        pH = torch.from_numpy(np.fromfile(path+'/../hKvproj/'+file,
                            dtype=np.float32)).view(1,1,ct.hsp['views'],ct.hsp['bins'])
        p_gt = -torch.log(torch.cat([pL,pH],dim=1).cuda())
        p_err = NNproj - p_gt
        # plt.figure(1)
        # plt.imshow(NNproj[0,0,...].data.cpu().numpy(),cmap='gray')
        # plt.figure(2)
        # plt.imshow(p_gt[0,0,...].data.cpu().numpy(),cmap='gray')
        # plt.show()
        recon_err = ct.ART2d_forward(p_err)
        recon_err.data.cpu().numpy().tofile(path+'/../err_0/'+file)
        p_err.data.cpu().numpy().tofile(path+'/../err_p/'+file)
        if train:
            A = torch.from_numpy(np.fromfile(path + '/../A/' + file,
                                dtype=np.float32)).view(1, 1, ct.rp['nSize'], ct.rp['nSize'])
            C = torch.from_numpy(np.fromfile(path + '/../C/' + file,
                                dtype=np.float32)).view(1, 1, ct.rp['nSize'], ct.rp['nSize'])
            F = torch.from_numpy(np.fromfile(path + '/../F/' + file,
                                dtype=np.float32)).view(1, 1, ct.rp['nSize'], ct.rp['nSize'])
            gt = torch.cat([A,C,F],dim=1).cuda()
            gt_err = NNrecon - gt
            gt_err.data.cpu().numpy().tofile(path+'/../err_gt/'+file)




path = 'F:/2022/dl_spectral_ct/ppt/'
# residualRecon(path+'/train/Model_0',True)
residualRecon(path+'/start_kit/Model_0',True)
# residualRecon(path+'/val/Model_0',False)

