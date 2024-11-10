import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.transforms import functional
import argparse
import time
from module import network
from utils import logger
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING']='1'

def getArgs():
    args = argparse.ArgumentParser()

    args.add_argument('--insize',default=1024,type=int)
    args.add_argument('--size',default=512,type=int)
    args.add_argument('--datapath',type=str,default=r'D:\GM0322\AAPM\label')
    args.add_argument('--gpu_device',type=list,default=[2,1])
    args.add_argument('--material',type=int,default=2)
    args.add_argument('--lr',type=float,default=1e-4)
    args.add_argument('--batch_size',type=int,default=9)
    args.add_argument('--epoch',type=int,default=1000)
    return args.parse_args()

def transform(input,target):
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

class eartdata(data.Dataset):
    def __init__(self,path,gpu_id,material=0,train=True,in_size=1024,size=512):
        self.path = path
        self.files = os.listdir(path)
        self.insize = in_size
        self.size = size
        self.train = train
        self.gpu_id = gpu_id
        self.material = material
        self.number = min(1000, len(self.files))
        self.input = torch.zeros((self.number, 2, in_size, in_size), dtype=torch.float32)
        if train:
            self.target = torch.zeros((self.number, 4, size, size), dtype=torch.float32)
        self.load_data()
        self.cuda(gpu_id)

    def __getitem__(self, item):
        input = self.input[item, ...].cuda(self.gpu_id)
        if self.train:
            target = self.target[item, self.material:self.material+1, ...].cuda(self.gpu_id)
            input, target = transform(input, target)
        else:
            target = 1
        return input, target, self.files[item]

    def __len__(self):
        return self.number  # len(self.files)

    def load_data(self):
        for i in range(self.number):
            file = self.files[i]
            self.input[i,...] = torch.from_numpy(np.fromfile(self.path+'/'+file,
                                                dtype=np.float32)).view(2,self.insize,self.insize).permute(0,2,1)
            if self.train:
                self.target[i,0,...] = torch.from_numpy(np.fromfile(self.path+'/../A/'+file,
                                                        dtype=np.float32)).view(1,self.size,self.size)
                self.target[i,1,...] = torch.from_numpy(np.fromfile(self.path+'/../C/'+file,
                                                        dtype=np.float32)).view(1,self.size,self.size)
                self.target[i,2,...] = torch.from_numpy(np.fromfile(self.path+'/../F/'+file,
                                                        dtype=np.float32)).view(1,self.size,self.size)
                self.target[i,3,...] = self.target[i,0:3,...].sum(dim=0, keepdim=True)

    def cuda(self,gpu_ids):
        self.input = self.input.cuda(gpu_ids)
        if self.train:
            self.target = self.target.cuda(gpu_ids)

def eart2gtLoader(path,batch_size=1,shuffle=True,train=True,material=0,insize=1024,size=512,gpu_id=0):
    dataset = eartdata(path,gpu_id=gpu_id,train=train,material=material,in_size=insize,size=size)
    return data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

def train():
    args = getArgs()
    print(args)
    train_loader = eart2gtLoader(args.datapath + '/train/first_eart2m2', args.batch_size,insize=args.insize,
                                 size=args.size,shuffle=True, gpu_id=args.gpu_device[0], material=args.material)
    start_loader = eart2gtLoader(args.datapath + '/starting_kit/first_eart2m2', 1,insize=args.insize,size=args.size,
                                 shuffle=False, gpu_id=args.gpu_device[0], material=args.material)
    model_name = 'Model_0_{}'.format(args.material)
    log = logger.Logger('./checkpoint/'+model_name+'.txt')
    model = network.network().cuda(args.gpu_device[0])
    model.load_state_dict(torch.load('./checkpoint/' + model_name + '_param.pt', map_location='cuda:{}'.format(args.gpu_device[0])))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='sum')

    # train network and update parameters
    for epoch in range(args.epoch):
        s = time.time()
        model.train()
        for step,(x,y,file) in enumerate(train_loader):
            x = x.cuda(args.gpu_device[0])
            y = y.cuda(args.gpu_device[0])
            out,out_ = model(x)
            loss = criterion(out,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if(step%100 == 0):
                print('epoch:{:03d},step={:03d},loss={:.9f},time:{:.1f}'.format(epoch,step,loss,time.time()-s))
                out.data.cpu().numpy().tofile('./res/' + model_name + '_0_res.raw')
                y.data.cpu().numpy().tofile('./res/' + model_name + '_0_gt.raw')
                (out-y).data.cpu().numpy().tofile('./res/' + model_name + '_0_err.raw')
                x.data.cpu().numpy().tofile('./res/' + model_name + '_0_in.raw')

        model.eval()
        torch.save(model,'./checkpoint/' + model_name + '_model.pt')
        torch.save(model.state_dict(),'./checkpoint/' + model_name + '_param.pt')

        # test train data and start_kit data
        with torch.no_grad():
        #     mse_loss = 0.0
        #     max_loss = 0.0
        #     max_err = 0.0
        #     for step,(x,y,file) in enumerate(train_loader):
        #         y = y.cuda(args.gpu_device[0])
        #         out,out_ = model(x.cuda(args.gpu_device[0]))
        #         loss = torch.sqrt(torch.mean(((out-y)**2))).item()
        #         mse_loss += loss
        #         if(max_loss<loss):
        #             max_loss = loss
        #         err = torch.max(torch.abs(out-y))
        #         if(max_err<err):
        #             max_err = err
        #         # if(step>9):
        #         #     break
        #     log.append('train:epoch:{:03d},mse:{:.9f},max:{:.9f},err:{:.3f},time:{:.1f}'.format(
        #         epoch,mse_loss/train_loader.__len__(),max_loss,max_err,time.time()-s))
        #     out.data.cpu().numpy().tofile('./res/' + model_name + '_1_res.raw')
        #     y.data.cpu().numpy().tofile('./res/' + model_name + '_1_gt.raw')
        #     (out - y).data.cpu().numpy().tofile('./res/' + model_name + '_1_err.raw')
            mse_loss = 0.0
            max_loss = 0.0
            max_err = 0.0
            for step,(x,y,file) in enumerate(start_loader):
                y = y.cuda(args.gpu_device[0])
                out,out_ = model(x.cuda(args.gpu_device[0]))
                loss = torch.sqrt(torch.mean(((out-y)**2))).item()
                mse_loss += loss
                if(max_loss<loss):
                    max_loss = loss
                err = torch.sqrt(torch.max(torch.abs(out-y)))
                if(max_err<err):
                    max_err = err
            log.append('test:epoch:{:03d},mse:{:.9f},max:{:.9f},err:{:.3f},time:{:.1f}'.format(
                epoch,mse_loss/start_loader.__len__(),max_loss,max_err,time.time()-s))
            out.data.cpu().numpy().tofile('./res/' + model_name + '_2_res.raw')
            y.data.cpu().numpy().tofile('./res/' + model_name + '_2_gt.raw')
            (out - y).data.cpu().numpy().tofile('./res/' + model_name + '_2_err.raw')
    print(args)

def test_data():
    args = getArgs()
    model_name = 'Model_0_{}'.format(args.material)
    model = network.network().cuda(args.gpu_device[0])
    model.load_state_dict(torch.load('./checkpoint/' + model_name + '_param.pt', map_location='cuda:{}'.format(args.gpu_device[0])))
    os.makedirs(args.datapath + '/train/' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/starting_kit/' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/val/' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/test/' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/train/f_' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/starting_kit/f_' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/val/f_' + model_name, exist_ok=True)
    os.makedirs(args.datapath + '/test/f_' + model_name, exist_ok=True)

    train_loader = eart2gtLoader(args.datapath + '/train/first_eart2m2', batch_size=1, shuffle=False,insize=args.insize,
                                 size=args.size,train=False,gpu_id=args.gpu_device[0],material=args.material)
    start_loader = eart2gtLoader(args.datapath + '/starting_kit/first_eart2m2', batch_size=1, shuffle=False,insize=args.insize,
                                 size=args.size,train=False,gpu_id=args.gpu_device[0],material=args.material)
    val_loader = eart2gtLoader(args.datapath + '/val/first_eart2m2', batch_size=1, shuffle=False,insize=args.insize,
                               size=args.size,train=False, gpu_id=args.gpu_device[0],material=args.material)
    # test_loader = eart2gtLoader(args.datapath + '/test/first_eart2m2', batch_size=1, shuffle=False,insize=args.insize,
    #                            size=args.size,train=False, gpu_id=args.gpu_device[0],material=args.material)
    with torch.no_grad():
        for step, (x, y, file) in tqdm(enumerate(train_loader)):
            out,out_ = model(x.cuda(args.gpu_device[0]),False)
            out_.data.cpu().numpy().tofile(args.datapath + '/train/' + model_name + '/' + file[0])
            out.data.cpu().numpy().tofile(args.datapath + '/train/f_' + model_name + '/' + file[0])
        print('--------------------------train finish---------------------------------')
        for step, (x, y, file) in enumerate(start_loader):
            out,out_ = model(x.cuda(args.gpu_device[0]),False)
            out_.data.cpu().numpy().tofile(args.datapath + '/starting_kit/' + model_name + '/' + file[0])
            out.data.cpu().numpy().tofile(args.datapath + '/starting_kit/f_' + model_name + '/' + file[0])
        print('--------------------------starting_kit finish---------------------------------')
        for step, (x, y, file) in enumerate(val_loader):
            out,out_ = model(x.cuda(args.gpu_device[0]),False)
            out_.data.cpu().numpy().tofile(args.datapath + '/val/' + model_name + '/' + file[0])
            out.data.cpu().numpy().tofile(args.datapath + '/val/f_' + model_name + '/' + file[0])

        # for step, (x, y, file) in enumerate(test_loader):
        #     out_, out = model(x.cuda(args.gpu_device[0]),False)
        #     out.data.cpu().numpy().tofile(args.datapath + '/test/' + model_name + '/' + file[0])
        print('--------------------------test finish---------------------------------')


if __name__ == "__main__":
    # train()
    test_data()


