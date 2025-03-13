import torch.nn as nn
import torch.nn.functional as F

"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv_featmap_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.conv_featmap_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.conv_featmap_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )
    
    def forward(self, img):
        featmap_1 = self.conv_featmap_1(img)
        featmap_1_down = self.maxpool(featmap_1)

        featmap_2 = self.conv_featmap_2(featmap_1_down)
        featmap_2_down = self.maxpool(featmap_2)

        featmap_3 = self.conv_featmap_3(featmap_2_down)
        
        return featmap_3
"""

class Encoder_RRDB(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=16):
        super(Encoder_RRDB, self).__init__()
        self.conv_featmap = nn.Sequential(
            nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
        )
    
    def forward(self, img):
        featmap = self.conv_featmap(img)
        
        return featmap
    
    
class Encoder_LR_RRDB(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=16):
        super(Encoder_LR_RRDB, self).__init__()
        self.conv_featmap = nn.Sequential(
            nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
        )
    
    def forward(self, img):
        featmap = self.conv_featmap(img)
        
        return featmap
    
    
"default conv layer"
def default_conv(in_channels, out_channels, kernel_size, stride=1, bias = True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=(kernel_size//2), bias=bias)


"Channel Attention (CA) Layer"
class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer, is basical block in RCAN. One CA forms one RCAB(Residual Channel Attention Block).
    See figure 3 of original RCAN paper.
    Beware the CA Layer used in RCAN is actually same as the channel attention mechanism propsed in SENet(Squeeze-and-Excitation Networks).
    """
    def __init__(self, channel, reduction=16):
        """
        reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        """
        super(CALayer, self).__init__()
        # global average pooling(GAP): feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global average pooling(GAP), output size is 1 for each channel
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), # W_d in CA
                nn.ReLU(inplace=False), # ReLU in CA
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), # W_u in CA
                nn.Sigmoid() # sigmoid in CA
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # the x is the "feature maps over channels" in size C x H x W. The y now is actual the weights in size C x 1 x 1 which represents "channel statistics", 
        # it stands for how much "attention" expected to pay for each channel's feature map 
        return x * y


"Residual Channel Attention Block (RCAB)"
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB): There are several RCABs belong to one RG(Residual Group).
    See figure 4 of original RCAN paper
    """
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2): # conv --> ReLU --> conv
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: 
                modules_body.append(nn.ReLU(True))
        modules_body.append(CALayer(n_feat, reduction)) # use CA Layer

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) # conv --> ReLU --> conv --> CA/channel and spatial attention block
        #res = self.body(x).mul(self.res_scale)
        x = x + res # local skip link of RCAB
        return x


"Residual Group (RG)"
class ResidualGroup(nn.Module):
    """
    RG(Residual Group): There are several RGs belong to one RIR(Residual in Residual module).
    See upper figure in figure 2 of original RCAN paper
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, res_scale, n_rcablocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, res_scale=1) \
            for _ in range(n_rcablocks)]

        modules_body.append(conv(n_feat, n_feat, kernel_size)) # last conv after several RCABs as show in figure 2
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # RCAB_1 --> RCAB_2 --> ... --> RCAB_(n_rcablocks) --> conv
        x = x + res # short skip connection in RG
        return x
    

"Residual Channel Attention Network (RCAN)"
class Encoder_RCAN(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """
    def __init__(self, num_in_ch, n_resgroups=3, n_rcablocks=3, num_out_ch=64, num_feat=64, scale=1):
        super(Encoder_RCAN, self).__init__()
        conv = default_conv

        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = 16 # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        
        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # define commonly use head module
        modules_head = [conv(num_in_ch, num_feat, kernel_size)] # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define commonly use pretail module
        modules_pretail = [conv(num_feat, num_feat, kernel_size)]

        # define body module for different type of network. 
        # The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
        modules_body = [
                ResidualGroup(
                    conv, num_feat, kernel_size, reduction, res_scale=1, n_rcablocks=n_rcablocks) \
                for _ in range(n_resgroups)]
        

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = []
        """
        modules_tail.append(
                    Upsampler(scale, num_feat))
        """
        modules_tail.append(conv(num_feat, num_out_ch, kernel_size))
        modules_tail.append(nn.ReLU(inplace=True))

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.pretail = nn.Sequential(*modules_pretail)
        self.tail = nn.Sequential(*modules_tail)
        
    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us 
        """ x = self.sub_mean(x) """
        x = self.head(x) # input image goes through first conv layer
        x = self.body(x) # data goes through sveral ResidualGroups
        x = self.pretail(x)
        x = self.tail(x) # data goes through upsampling module and one more conv layer
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.add_mean(x) """
        return x
        
"Residual Channel Attention Network (RCAN)"
class Encoder_LR_RCAN(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """
    def __init__(self, num_in_ch, n_resgroups=3, n_rcablocks=3, num_out_ch=64, num_feat=64, scale=1):
        super(Encoder_LR_RCAN, self).__init__()
        conv = default_conv

        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = 16 # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        
        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # define commonly use head module
        modules_head = [conv(num_in_ch, num_feat, kernel_size, stride=2)] # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define commonly use pretail module
        modules_pretail = [conv(num_feat, num_feat, kernel_size)]

        # define body module for different type of network. 
        # The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
        modules_body = [
                ResidualGroup(
                    conv, num_feat, kernel_size, reduction, res_scale=1, n_rcablocks=n_rcablocks) \
                for _ in range(n_resgroups)]
        

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = []
        """
        modules_tail.append(
                    Upsampler(scale, num_feat))
        """
        modules_tail.append(conv(num_feat, num_out_ch, kernel_size))
        modules_tail.append(nn.ReLU(inplace=True))

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.pretail = nn.Sequential(*modules_pretail)
        self.tail = nn.Sequential(*modules_tail)
        
    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us 
        """ x = self.sub_mean(x) """
        x = self.head(x) # input image goes through first conv layer
        x = self.body(x) # data goes through sveral ResidualGroups
        x = self.pretail(x)
        x = self.tail(x) # data goes through upsampling module and one more conv layer
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.add_mean(x) """
        return x
    

    
class Encoder_RRDB_3D(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=16):
        super(Encoder_RRDB_3D, self).__init__()
        self.conv_featmap = nn.Sequential(
            nn.Conv3d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
        )
    
    def forward(self, img):
        featmap = self.conv_featmap(img)
        
        return featmap