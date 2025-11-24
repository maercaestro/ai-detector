import torchvision.models as models

class SSPDetector(nn.Module):
    def __init__(self):
        super(SSPDetector, self).__init__()
        # 1. Preprocessing Layer (SRM)
        self.srm = SRMLayer()
        
        # 2. Backbone (ResNet-50)
        # Note: We must modify the first layer because SRM output is 30 channels, not 3.
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify first conv layer to accept 30 channels
        # Standard ResNet: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, 
                                      padding=3, bias=False)
        
        # Initialize new weights (e.g., by averaging original weights or random init)
        # Here we use Kaiming initialization
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # 3. Classification Head (Binary: Real vs Fake)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        # x is the Simplest Patch (B, 3, 256, 256)
        
        # Extract Residuals
        residuals = self.srm(x) # Output: (B, 30, 256, 256)
        
        # Classify
        logits = self.resnet(residuals)
        return logits