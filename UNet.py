import torch

class StackDown(torch.nn.Module):
    def __init__(self,in_channels=1,out_channels=1):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
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
            torch.nn.ConvTranspose2d(hidden_channels,out_channels,kernel_size=3,stride=1,padding=0)
        )
    
    def forward(self, x):
        return self.layers(x)

class UNet(torch.nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.stack_1 = StackDown(1,64) # 572x572 -> 285x284
        self.stack_2 = StackDown(64,128) # 140x140
        self.stack_3 = StackDown(128,256) # 68x68
        self.stack_4 = StackDown(256,512) # 32x32

        self.bottom_conv_1 = torch.nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=0)
        self.bottom_conv_2 = torch.nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=0)