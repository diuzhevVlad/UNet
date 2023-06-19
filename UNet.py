import torch
from torchvision.transforms import CenterCrop
from torch.nn.functional import interpolate

class StackDown(torch.nn.Module):
    def __init__(self,in_channels=1,out_channels=1):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)
    
class StackUp(torch.nn.Module):
    def __init__(self,in_channels=1,hidden_channels=1,out_channels=1):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,hidden_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden_channels,out_channels,kernel_size=2,stride=2,padding=0)
        )
    
    def forward(self, to_crop, x):
        if to_crop is not None:
            (_, _, H, W) = x.shape
            cropped = CenterCrop([H, W])(to_crop)
            cat = torch.cat([cropped,x],1)
            return self.layers(cat)
        else:
            return self.layers(x)
    
class SegmentationHead(torch.nn.Module):
    def __init__(self,in_channels=1,hidden_channels=1,num_classes=2):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,hidden_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels,num_classes,kernel_size=1,stride=1,padding=0),
        )
    
    def forward(self, to_crop, x):
        if to_crop is not None:
            (_, _, H, W) = x.shape
            cropped = CenterCrop([H, W])(to_crop)
            cat = torch.cat([cropped,x],1)
            return self.layers(cat)
        else:
            return self.layers(x)
    

class UNet(torch.nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.poolling_list = torch.nn.ModuleList([
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.MaxPool2d(kernel_size=2)
        ])
        self.stack_1 = StackDown(3,64) 
        self.stack_2 = StackDown(64,128)
        self.stack_3 = StackDown(128,256)
        self.stack_4 = StackDown(256,512) 

        self.stack_5 = StackUp(512,1024,512) 
        self.stack_6 = StackUp(1024,512,256)
        self.stack_7 = StackUp(512,256,128)
        self.stack_8 = StackUp(256,128,64) 

        self.seg_head = SegmentationHead(128,64,num_classes) 

    def forward(self,x):
        st_1_res = self.stack_1(x)
        st_2_res = self.stack_2(self.poolling_list[0](st_1_res))
        st_3_res = self.stack_3(self.poolling_list[1](st_2_res))
        st_4_res = self.stack_4(self.poolling_list[2](st_3_res))

        st_5_res = self.stack_5(None, self.poolling_list[3](st_4_res))
        st_6_res = self.stack_6(st_4_res,st_5_res)
        st_7_res = self.stack_7(st_3_res,st_6_res)
        st_8_res = self.stack_8(st_2_res,st_7_res)

        classes_mask = self.seg_head(st_1_res,st_8_res)
        return interpolate(classes_mask,x.shape[2:])