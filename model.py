#imports
import torch
import torch.nn as nn
import torchvision.models as models  

#CNN that outputs x, y, w, h, sin(2*theta), cos(2*theta)
class GraspNet(nn.Module): 
    def __init__(self): 
        super().__init__()

        #load resnet18
        backbone = models.resnet18(weights = "IMAGENET1K_V1")

        #add 4th channel for depth
        #grab original first layer to copy its settings
        old_conv = backbone.conv1

        #build new conv layer 
        new_conv = nn.Conv2d(
            in_channels = 4, 
            out_channels = old_conv.out_channels, 
            kernel_size = old_conv.kernel_size, 
            stride = old_conv.stride, 
            padding = old_conv.padding, 
            bias = (old_conv.bias is not None)
        )

        #initialize new channel with red channel's weights 
        with torch.no_grad(): 
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight[:, 0 : 1]

        backbone.conv1 = new_conv

        #remove final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        #regression head (replacement for removed classification layer)
        self.head = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(512, 128),
            nn.ReLU(), 
            nn.Dropout(0.3), 
            #final layer: 128 in, 6 out
            nn.Linear(128, 6)
        )

    #data through the model
    def forward(self, x): 
        #ResNet layers and 512 number summary
        features = self.backbone(x)
        #summary in, final 6 numbers out 
        return self.head(features)

if __name__ == "__main__": 
    #test if model runs 
    model = GraspNet()
    #2 fake images 4 channels each 
    dummy_input = torch.randn(2, 4, 224, 224)
    output = model(dummy_input)
    #should print torch.Size([2, 6]), 2 in 6 nums out 
    print("Output shape:", output.shape)
            
        