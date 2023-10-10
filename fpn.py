import torch.nn as nn
import torch
from torchvision.models import resnet34, ResNet34_Weights

model = resnet34(weights=ResNet34_Weights.DEFAULT)


##### building fpn network ##### 

class FPN(nn.Module):
  def __init__(self,model,channel):
    super(FPN,self).__init__()
    self.model = model
    self.conv1 = nn.Sequential(nn.Conv2d(channel,64,7,2,3),
                               nn.BatchNorm2d(64),
                               nn.ReLU(inplace=True),
                               nn.MaxPool2d(3,2,1))
    
    self.out4 = nn.Conv2d(512,64,1)
    self.out3 = nn.Conv2d(256,64,1)
    self.out2 = nn.Conv2d(128,64,1)
    self.out1 = nn.Conv2d(64,64,1)

    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
  

  def forward(self,x):
    conv1 = self.conv1(x)
    out1 = self.model.layer1(conv1)
    print(f'Conv1 shape : {out1.shape}')
    out2 = self.model.layer2(out1)
    print(f'Conv2 shape : {out2.shape}')
    out3 = self.model.layer3(out2)
    print(f'Conv3 shape : {out3.shape}')
    out4 = self.model.layer4(out3)
    print(f'Conv4 shape : {out4.shape}')


    print('--Changing filters--')
    # add out4 output to out3
    # chnage out4 filter size
    
    out4 = self.out4(out4) # first o1
    out3 = self.out3(out3)
    out2 = self.out2(out2)
    out1 = self.out1(out1)

    print(f'out4 shape : {out4.shape}')
    print(f'out3 shape : {out3.shape}')
    print(f'out2 shape : {out2.shape}')
    print(f'out1 shape : {out1.shape}')
    


    # upsampling
    print(f'--upsampling out4--')
    out4_upsample = self.upsample(out4)
    
    print('adding the upsample out4 (upsampled) with out3')

    out3_and_out4_add = torch.add(out3,out4_upsample) # 2nd pred
    print(f'added out3 and out4_up : {out3_and_out4_add.shape}')

    print(f'upsample out3 and out4_up...')
    out3_and_out4_upsample = self.upsample(out3_and_out4_add)

    out2_out3_and_out4_add = torch.add(out3_and_out4_upsample,out2) #3rd pred
    print(f'added out2 out3 and out4_up : {out2_out3_and_out4_add.shape}')

    print(f'upsample out2 out3 and out4_up...')
    out2_out3_and_out4_upsample = self.upsample(out2_out3_and_out4_add)

    out1_out2_out3_and_out4_add = torch.add(out2_out3_and_out4_upsample,out1) # 4th pred
    print(f'out1_out2_out3_and_out4_add : {out1_out2_out3_and_out4_add.shape}')

    return out4 , out3_and_out4_add , out2_out3_and_out4_add , out1_out2_out3_and_out4_add




if __name__ == '__main__':
    fpn = FPN(model,3)
    #testing
    image = torch.randn(1,3,224,224)
    out1 , out2 , out3 , out4 = fpn(image)

    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)