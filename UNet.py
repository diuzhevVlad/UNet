import torch

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
    
    def forward(self, x):
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
            # torch.nn.Softmax(1)
        )
    
    def forward(self, x):
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
        self.stack_1 = StackDown(1,64) # 568x568
        self.stack_2 = StackDown(64,128) # 280x280
        self.stack_3 = StackDown(128,256) # 136x136
        self.stack_4 = StackDown(256,512) # 64x64

        self.stack_5 = StackUp(512,1024,512) # 56x56
        self.stack_6 = StackUp(1024,512,256) # 104x104
        self.stack_7 = StackUp(512,256,128) # 200x200
        self.stack_8 = StackUp(256,128,64) # 392x392

        self.seg_head = SegmentationHead(128,64,num_classes) # 388x388

    def forward(self,x):
        st_1_res = self.stack_1(x)
        st_2_res = self.stack_2(self.poolling_list[0](st_1_res))
        st_3_res = self.stack_3(self.poolling_list[1](st_2_res))
        st_4_res = self.stack_4(self.poolling_list[2](st_3_res))

        st_5_res = self.stack_5(self.poolling_list[3](st_4_res))
        st_6_res = self.stack_6(torch.cat([st_4_res[:,:,4:-4,4:-4],st_5_res],1))
        st_7_res = self.stack_7(torch.cat([st_3_res[:,:,16:-16,16:-16],st_6_res],1))
        st_8_res = self.stack_8(torch.cat([st_2_res[:,:,40:-40,40:-40],st_7_res],1))

        return self.seg_head(torch.cat([st_1_res[:,:,88:-88,88:-88],st_8_res],1))