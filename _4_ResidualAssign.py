import numpy as np
import os
import torch
from module import operator,network
import argparse
import torch.utils.data as data
from torchvision.transforms import functional
import time
from utils import logger
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 设置为GPU编号为0的设备

def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--isGetResidualData',type=bool,default=False)
    args.add_argument('--numberOfIter',type=int,default=0)
    args.add_argument('--datapath',type=str,default='D:\GM0322\AAPM')
    args.add_argument('--gpu_device',type=list,default=[3,1])
    args.add_argument('--material',type=int,default=2)
    args.add_argument('--size',type=int,default=512)
    args.add_argument('--batch_size',type=int,default=100)
    args.add_argument('--lr',type=float,default=1e-4)
    args.add_argument('--epoch',type=int,default=1000)
    return args.parse_args()

def residualRecon(path,numberOfIter,train=True):
    ct = operator.CT()
    files = os.listdir(path)
    os.makedirs(path+'/../err_{}'.format(numberOfIter),exist_ok=True)
    if train:
        os.makedirs(path+'/../err_{}_gt'.format(numberOfIter),exist_ok=True)
    for i,file in enumerate(files):
        A = torch.from_numpy(np.fromfile(path+'/../f_Model_{}_0/'.format(numberOfIter)+file,
                            dtype=np.float32)).view(1,1,ct.rp['nSize'],ct.rp['nSize'])
        C = torch.from_numpy(np.fromfile(path+'/../f_Model_{}_1/'.format(numberOfIter)+file,
                            dtype=np.float32)).view(1,1,ct.rp['nSize'],ct.rp['nSize'])
        F = torch.from_numpy(np.fromfile(path+'/../f_Model_{}_2/'.format(numberOfIter)+file,
                            dtype=np.float32)).view(1,1,ct.rp['nSize'],ct.rp['nSize'])
        NNrecon = torch.cat([A,C,F],dim=1).cuda()
        NNproj = ct.spectrum_proj_forward(NNrecon)
        pL = torch.from_numpy(np.fromfile(path+'/../lKvproj/'+file,
                            dtype=np.float32)).view(1,1,ct.lsp['views'],ct.lsp['bins'])
        pH = torch.from_numpy(np.fromfile(path+'/../hKvproj/'+file,
                            dtype=np.float32)).view(1,1,ct.hsp['views'],ct.hsp['bins'])
        p_gt = -torch.log(torch.cat([pL,pH],dim=1).cuda())
        p_err = NNproj - p_gt

        recon_err = ct.ART2d_forward(p_err)
        recon_err.data.cpu().numpy().tofile(path+'/../err_{}/'.format(numberOfIter)+file)
        if train:
            A = torch.from_numpy(np.fromfile(path + '/../A/' + file,
                                dtype=np.float32)).view(1, 1, ct.rp['nSize'], ct.rp['nSize'])
            C = torch.from_numpy(np.fromfile(path + '/../C/' + file,
                                dtype=np.float32)).view(1, 1, ct.rp['nSize'], ct.rp['nSize'])
            F = torch.from_numpy(np.fromfile(path + '/../F/' + file,
                                dtype=np.float32)).view(1, 1, ct.rp['nSize'], ct.rp['nSize'])
            gt = torch.cat([A,C,F],dim=1).cuda()
            gt_err = NNrecon - gt
            gt_err.data.cpu().numpy().tofile(path+'/../err_{}_gt/'.format(numberOfIter)+file)

def  transform(input,target):
    '''
    左右，上下，主对角对称，次对角对称，旋转90，180，270，360
    '''
    type = np.random.randint(8)
    if type == 0:
        input = functional.hflip(input).contiguous()
        target = functional.hflip(target).contiguous()
    if type == 1:
        input = functional.hflip(input).contiguous()
        target = functional.hflip(target).contiguous()
    if type == 2:
        input = input.permute(0,2,1).contiguous()
        target = target.permute(0,2,1).contiguous()
    if type == 3:
        input = functional.hflip(input)
        target = functional.hflip(target)
        input = input.permute(0,2,1).contiguous()
        target = target.permute(0,2,1).contiguous()
    if type == 4:
        input = functional.rotate(input,90)
        target = functional.rotate(target,90)
    if type == 5:
        input = functional.rotate(input,180)
        target = functional.rotate(target,180)
    if type == 6:
        input = functional.rotate(input,270)
        target = functional.rotate(target,270)
    if type == 7:
        input = functional.rotate(input,360)
        target = functional.rotate(target,360)
    return input,target

class residualData(data.Dataset):
    def __init__(self,path,train=True,gpu_id=0,material=0,numberofIter=0,size=512):
        self.path = path
        self.files = os.listdir(path)
        self.train = train
        self.size = size
        self.material = material
        self.gpu_id = gpu_id
        self.numberofIter = numberofIter
        self.input = torch.zeros((len(self.files),5,size,size),dtype=torch.float32).cuda(gpu_id)
        if train:
            self.target = torch.zeros((len(self.files),3,size,size),dtype=torch.float).cuda(gpu_id)
        self.load_data()

    def __getitem__(self, item):
        input = self.input[item, ...].cuda(self.gpu_id)
        if self.train:
            target = self.target[item, self.material:self.material + 1, ...].cuda(self.gpu_id)
            target = self.target[item, ...].cuda(self.gpu_id).sum(dim=0,keepdim=True)
            input, target = transform(input, target)
        else:
            target = 1
        return input, target, self.files[item]

    def __len__(self):
        return len(self.files)

    def load_data(self):
        for i,file in enumerate(self.files):
            if self.train:
                input = torch.from_numpy(np.fromfile(self.path+'/../err_{}_block/'.format(self.numberofIter)+file,
                                                     dtype=np.float32)).view(5,self.size,self.size).cuda(self.gpu_id)
                self.input[i,...] = input
                target = torch.from_numpy(np.fromfile(self.path + '/../err_{}_gt_block/'.format(self.numberofIter) + file,
                                                      dtype=np.float32)).view(3, self.size, self.size).cuda(self.gpu_id)
                self.target[i, ...] = target
            else:
                input = torch.from_numpy(np.fromfile(self.path + '/../err_{}/'.format(self.numberofIter) + file,dtype=np.float32)).view(
                    2, self.size, self.size).permute(0, 2, 1).cuda(self.gpu_id)
                # input[torch.abs(input) < 6e-4] = 0
                self.input[i, 0:2, ...] = input
                self.input[i, 2, ...] = torch.from_numpy(
                    np.fromfile(self.path + '/../f_Model_{}_0/'.format(self.numberofIter) + file,
                                dtype=np.float32)).view(self.size, self.size).cuda(self.gpu_id)
                self.input[i, 3, ...] = torch.from_numpy(
                    np.fromfile(self.path + '/../f_Model_{}_1/'.format(self.numberofIter) + file,
                                dtype=np.float32)).view(self.size, self.size).cuda(self.gpu_id)
                self.input[i, 4, ...] = torch.from_numpy(
                    np.fromfile(self.path + '/../f_Model_{}_2/'.format(self.numberofIter) + file,
                                dtype=np.float32)).view(self.size, self.size).cuda(self.gpu_id)

def residualLoader(path,batch_size=1,shuffle=True,train=True,material=0,size=512,gpu_id=0,numberofIter=0):
    dataset = residualData(path,gpu_id=gpu_id,train=train,material=material,size=size,numberofIter=numberofIter)
    return data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

def train():
    args = getArgs()
    print(args)
    train_loader = residualLoader(args.datapath + '/train/err_{}_block'.format(args.numberOfIter), args.batch_size, shuffle=True,
                                  train=True,gpu_id=args.gpu_device[0], material=args.material,size=32,numberofIter=args.numberOfIter)
    start_loader = residualLoader(args.datapath + '/starting_kit/err_{}_block'.format(args.numberOfIter), 1, shuffle=True,
                                  train=True,gpu_id=args.gpu_device[0], material=args.material,size=32,numberofIter=args.numberOfIter)
    model_name = 'Model_{}_{}'.format(args.numberOfIter+1,args.material)
    log = logger.Logger('./checkpoint/'+model_name+'.txt')
    model = network.resassignnetwork(in_channel=6, out_channel=1).cuda(args.gpu_device[0])
    model.load_state_dict(torch.load('./checkpoint/' + model_name + '_param.pt', map_location='cuda:{}'.format(args.gpu_device[0])))

    optim = torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='sum')

    for epoch in range(args.epoch):
        s = time.time()
        model.train()
        for step,(x,y,file) in enumerate(train_loader):
            x = x.cuda(args.gpu_device[0])
            x1 = x[:, 0:1, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x2 = x[0, 1:2, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x = torch.cat([x1, x2], dim=1)
            y = y.cuda(args.gpu_device[0])
            out,out_ = model(x)
            loss = criterion(out_,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if(step%100 == 0):
                print('epoch:{:03d},step={:03d}，loss={:.9f},time:{:.1f},maxValue:{}'.format(epoch,step,loss,time.time()-s,out_.max()))
                out.data.cpu().numpy().tofile('./res/' + model_name + '_0_res.raw')
                y.data.cpu().numpy().tofile('./res/' + model_name + '_0_gt.raw')
                (out-y).data.cpu().numpy().tofile('./res/' + model_name + '_0_err.raw')
                x.data.cpu().numpy().tofile('./res/' + model_name + '_0_in.raw')

        model.eval()
        torch.save(model,'./checkpoint/' + model_name + '_model.pt')
        torch.save(model.state_dict(),'./checkpoint/' + model_name + '_param.pt')

        with torch.no_grad():
            # mse_loss = 0.0
            # max_loss = 0.0
            # max_err = 0.0
            # for step,(x,y,file) in enumerate(train_loader):
            #     y = y.cuda(args.gpu_device[0])
            #     out,out_ = model(x.cuda(args.gpu_device[0]))
            #     loss = torch.sqrt(torch.mean(((out_-y)**2))).item()
            #     mse_loss += loss
            #     if(max_loss<loss):
            #         max_loss = loss
            #     err = torch.max(torch.abs(out-y))
            #     if(max_err<err):
            #         max_err = err
            #     # if(step>9):
            #     #     break
            # log.append('train:epoch:{:03d},mse:{:.9f},max:{:.9f},err:{:.3f},time:{:.1f}'.format(
            #     epoch,mse_loss/train_loader.__len__(),max_loss,max_err,time.time()-s))
            # out.data.cpu().numpy().tofile('./res/' + model_name + '_1_res.raw')
            # y.data.cpu().numpy().tofile('./res/' + model_name + '_1_gt.raw')
            # (out - y).data.cpu().numpy().tofile('./res/' + model_name + '_1_err.raw')
            mse_loss = 0.0
            max_loss = 0.0
            max_err = 0.0
            for step,(x,y,file) in enumerate(start_loader):
                x = x.cuda(args.gpu_device[0])
                x1 = x[:, 0:1, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
                x2 = x[0, 1:2, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
                x = torch.cat([x1, x2], dim=1)
                y = y.cuda(args.gpu_device[0])
                out,out_ = model(x.cuda(args.gpu_device[0]))
                loss = torch.sqrt(torch.mean(((out_-y)**2))).item()
                mse_loss += loss
                if(max_loss<loss):
                    max_loss = loss
                err = torch.sqrt(torch.max(torch.abs(out-y)))
                if(max_err<err):
                    max_err = err
            log.append('test:epoch:{:03d},mse:{:.9f},max:{:.9f},err:{:.3f},time:{:.1f},maxValue:{:.2f}'.format(
                epoch,mse_loss/start_loader.__len__(),max_loss,max_err,time.time()-s,out.max()))
            out_.data.cpu().numpy().tofile('./res/' + model_name + '_2_res.raw')
            y.data.cpu().numpy().tofile('./res/' + model_name + '_2_gt.raw')
            (out_ - y).data.cpu().numpy().tofile('./res/' + model_name + '_2_err.raw')
    print(args)
def data_test():
    args = getArgs()
    model_name = 'Model_{}_{}'.format(args.numberOfIter+1,args.material)
    model = network.resassignnetwork(in_channel=6, out_channel=1).cuda(args.gpu_device[0])
    model.load_state_dict(torch.load('./checkpoint/' + model_name + '_param.pt', map_location='cuda:{}'.format(args.gpu_device[0])))


    os.makedirs(args.datapath + '/train/' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/starting_kit/' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/val/' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/train/f_' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/starting_kit/f_' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/val/f_' + model_name, exist_ok=True)

    train_loader = residualLoader(args.datapath + '/train/err_{}'.format(args.numberOfIter), 1, shuffle=False,
                                  train=False, gpu_id=args.gpu_device[0], material=args.material,
                                  numberofIter=args.numberOfIter)
    start_loader = residualLoader(args.datapath + '/starting_kit/err_{}'.format(args.numberOfIter), 1, shuffle=False,
                                  train=False,gpu_id=args.gpu_device[0], material=args.material,
                                  numberofIter=args.numberOfIter)
    val_loader = residualLoader(args.datapath + '/val/err_{}'.format(args.numberOfIter), 1, shuffle=False,
                                train=False,gpu_id=args.gpu_device[0],material=args.material,
                                numberofIter=args.numberOfIter)

    with torch.no_grad():
        for step, (x, y, file) in enumerate(train_loader):
            x = x.cuda(args.gpu_device[0])
            x1 = x[:, 0:1, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x2 = x[0, 1:2, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x = torch.cat([x1, x2], dim=1)
            out,out_ = model(x.cuda(args.gpu_device[0]))
            out.data.cpu().numpy().tofile(args.datapath + '/train/' + model_name + '/' + file[0])
            out_.data.cpu().numpy().tofile(args.datapath + '/train/f_' + model_name + '/' + file[0])
        for step, (x, y, file) in enumerate(start_loader):
            x = x.cuda(args.gpu_device[0])
            x1 = x[:, 0:1, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x2 = x[0, 1:2, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x = torch.cat([x1, x2], dim=1)
            out,out_ = model(x.cuda(args.gpu_device[0]))
            out.data.cpu().numpy().tofile(args.datapath + '/starting_kit/' + model_name + '/' + file[0])
            out_.data.cpu().numpy().tofile(args.datapath + '/starting_kit/f_' + model_name + '/' + file[0])

        for step, (x, y, file) in enumerate(val_loader):
            x = x.cuda(args.gpu_device[0])
            x1 = x[:, 0:1, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x2 = x[0, 1:2, ...] * (1 - 2 * torch.abs(x[:, 2:, ...] - 0.5))
            x = torch.cat([x1, x2], dim=1)
            out,out_ = model(x.cuda(args.gpu_device[0]))
            out.data.cpu().numpy().tofile(args.datapath + '/val/' + model_name + '/' + file[0])
            out_.data.cpu().numpy().tofile(args.datapath + '/val/f_' + model_name + '/' + file[0])



# if __name__ == '__main__':
args = getArgs()
if args.isGetResidualData:
    residualRecon(args.datapath + '/train/Model_{}_0'.format(args.numberOfIter),args.numberOfIter, True)
    residualRecon(args.datapath + '/starting_kit/Model_{}_0'.format(args.numberOfIter),args.numberOfIter, True)
    residualRecon(args.datapath + '/val/Model_{}_0'.format(args.numberOfIter),args.numberOfIter, False)
train()
data_test()


























