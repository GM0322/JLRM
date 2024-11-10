from torch.utils.dlpack import to_dlpack,from_dlpack
from utils import config
import cupy
from module import EART,ART,FanBeamProjection
import torch

class CT():
    def __init__(self):
        self.rp,self.lsp,self.hsp = config.ScanParam()
        args = config.getArgs()
        self.sl,self.sh,self.muH2O,self.muAl,self.muBone = config.load_spectrum_and_mu(args.low_spectrum,args.high_spectrum)

    def EART2S2M_forward(self,proj,xinit=None):
        B = proj.shape[0]
        cuproj = cupy.from_dlpack(to_dlpack(proj.detach()))
        if xinit is None:
            cuxinit = cupy.zeros((B,2,self.rp['nSize'],self.rp['nSize']),dtype=cupy.float32)
        else:
            cuxinit = cupy.from_dlpack(to_dlpack(xinit.detach()))
        x1 = cupy.zeros_like(cuxinit,dtype=cupy.float32)
        for i in range(B):
            x1[i,...] = EART.EART2S2M(cuxinit[i,...],cuproj[i,...], self.rp, self.lsp, self.hsp, self.muH2O, self.muAl, self.sl, self.sh, 1)
        return from_dlpack(x1.toDlpack())

    def EART2S3M_forward(self,proj,xinit):
        B = proj.shape[0]
        cuproj = cupy.from_dlpack(to_dlpack(proj.detach()))
        cuxinit = cupy.from_dlpack(to_dlpack(xinit.detach()))
        x1 = EART.EART2S3M(cuxinit[0, ...], cuproj[0, ...], self.rp, self.lsp, self.hsp, self.muH2O, self.muAl, self.muBone, self.sl, self.sh, 1).unsqueeze(0)
        for i in range(1,B):
            x11 = EART.EART2S3M(cuxinit[i,...], cuproj[i,...], self.rp, self.lsp, self.hsp, self.muH2O, self.muAl, self.muBone, self.sl, self.sh, 1).unsqueeze(0)
            x1 = torch.cat([x1,x11],dim=0)
        return from_dlpack(x1.toDlpack())

    def projection_forward(self,x,sp):
        B,C,H,W = x.shape
        cuproj = cupy.zeros((B,C,sp['views'],sp['bins']),dtype=cupy.float32)
        cux = cupy.from_dlpack(to_dlpack(x.detach()))
        for i in range(B):
            for j in range(C):
                cuproj[i,j,...] = FanBeamProjection.projection(cux[i,j,...],sp,self.rp)
        return from_dlpack(cuproj.toDlpack())

    def spectrum_proj_forward(self,x):
        p = self.projection_forward(x,self.lsp)
        B, C, H, W = p.shape
        assert C == 3, 'input error of function "spectrum_proj_forward" '
        spect_proj = torch.zeros((B,2,H,W),dtype=torch.float32).cuda(x.device)
        sl = cupy.asnumpy(self.sl)
        sh = cupy.asnumpy(self.sh)
        muH2O = cupy.asnumpy(self.muH2O)
        muAl = cupy.asnumpy(self.muAl)
        muBone = cupy.asnumpy(self.muBone)
        for i in range(sl.shape[0]):
            spect_proj[:,0,...] += 0.5*sl[i]*torch.exp(-muH2O[i]*p[:,0,...]-muBone[i]*p[:,1,...]-muAl[i]*p[:,2,...])
        p = self.projection_forward(x,self.hsp)
        assert p.shape[1] == 3, 'input error of function "spectrum_proj_forward" '
        for i in range(sh.shape[0]):
            spect_proj[:,1,...] += 0.5*sh[i]*torch.exp(-muH2O[i]*p[:,0,...]-muBone[i]*p[:,1,...]-muAl[i]*p[:,2,...])
        return -torch.log(spect_proj)

    def ART2d_forward(self,p,xinit=None):
        assert p.shape[1] == 2, 'input error of function "art2d_forward" '
        if xinit is None:
            cu_xinit = cupy.zeros((p.shape[0],2, self.rp['nSize'],self.rp['nSize']),dtype=cupy.float32)
        else:
            cu_xinit = cupy.from_dlpack(to_dlpack(xinit))
        cuproj = cupy.from_dlpack(to_dlpack(p.detach()))
        curecon = cupy.zeros_like(cu_xinit,dtype=cupy.float32)
        for i in range(p.shape[0]):
            curecon[i,0,...] = ART.ART2D(cuproj[i,0,...],self.lsp,self.rp,cu_xinit[i,0,...])
            curecon[i,1,...] = ART.ART2D(cuproj[i,1,...],self.hsp,self.rp,cu_xinit[i,1,...])
        return from_dlpack(curecon.toDlpack())
