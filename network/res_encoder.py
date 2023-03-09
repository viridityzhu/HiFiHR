import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
from torchvision import models
import timm


class ResEncoder(nn.Module):
    def __init__(self, nc:int=4, nk:int=1, pretrain:str='hr18sv2', droprate=0.0, coordconv=False, norm = 'bn', if_4c=True):
        super(ResEncoder, self).__init__()
        self.mmpool = MMPool((1,1))
        if pretrain=='none':
            # 2-4-4-3 = 12 resblocks = 24 conv
            self.encoder1 = Base_4C(nc=nc, nk=nk, norm = norm, coordconv=coordconv)
            self.encoder1.apply(weights_init)
            in_dim = 288 
        elif pretrain=='unet': #unet from scratch 
            self.encoder1 = UNet_4C(nc=nc, nk=nk, norm = norm, coordconv=coordconv)
            self.encoder1.apply(weights_init)
            in_dim = 32
        elif pretrain=='res18':
            self.encoder1 = Resnet_4C(pretrain)
            in_dim = 512 
        elif pretrain=='res50':
            self.encoder1 = Resnet_4C(pretrain)
            in_dim = 2048 
        elif 'hr18' in pretrain:
            self.encoder1 = HRnet_4C(pretrain, if_4c=if_4c)
            in_dim = 2048 
        else: 
            print('unknown network')
        if if_4c:
            self.norm_func = normalize_batch_4C
        else:
            self.norm_func = normalize_batch_3C

    def forward(self, x):
        ################### PreProcessing
        x = self.norm_func(x) 
        #################### Backbone
        #with torch.no_grad():
        low_features, features = self.encoder1(x)  # [b, 512, 14, 14] = 100352, 4: [b, 1024, 7, 7] = 50176
        features = self.mmpool(features).view(features.shape[0], -1) # [b, 1024, 7, 7] -> [b, 1024, 1, 1] -> [b, 1024]
        return low_features, features 


class HandEncoder(nn.Module):
    '''
        Estimates:
            joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses
    '''
    def __init__(self, ncomps:list, in_dim=1024, use_mean_shape=False, ifRender=True):
        super(HandEncoder, self).__init__()
        if use_mean_shape:
            print("use mean MANO shape")
        else:
            print("do not use mean MANO shape")
        self.use_mean_shape = use_mean_shape
        self.ifRender = ifRender
        self.shape_ncomp, self.pose_ncomp, self.tex_ncomp = ncomps

        # Base layers: in_dim -> 512
        base_layers = []
        base_layers.append(nn.Linear(in_dim, 1024))
        base_layers.append(nn.BatchNorm1d(1024))
        base_layers.append(nn.ReLU(inplace=True))
        base_layers.append(nn.Linear(1024, 512))
        base_layers.append(nn.BatchNorm1d(512))
        base_layers.append(nn.ReLU(inplace=True))
        self.base_layers = nn.Sequential(*base_layers)
        self.base_layers.apply(weights_init)

        # Pose Layers: 512 -> 30
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(128, self.pose_ncomp))
        self.pose_reg = nn.Sequential(*layers)
        self.pose_reg.apply(weights_init)

        # Shape Layers: 512 -> 20
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(128, self.shape_ncomp))
        self.shape_reg = nn.Sequential(*layers)
        self.shape_reg.apply(weights_init)

        # Texture Layers: 512 -> 10
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(128, self.tex_ncomp))
        self.tex_reg = nn.Sequential(*layers)
        self.tex_reg.apply(weights_init)

        # Trans layers: 512 -> 3
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 3))
        self.trans_reg = nn.Sequential(*layers)
        self.trans_reg.apply(weights_init)

        # # rot layers: 512 -> 3
        # layers = []
        # layers.append(nn.Linear(512, 128))
        # layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(128, 32))
        # layers.append(nn.Linear(32, 3))
        # self.rot_reg = nn.Sequential(*layers)
        # self.rot_reg.apply(weights_init)

        # scale layers: 512 -> 1
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 1))
        self.scale_reg = nn.Sequential(*layers)
        self.scale_reg.apply(weights_init)

    def forward(self, features):
        bs = features.shape[0]
        device = features.device
        
        base_features = self.base_layers(features)
        pose_params = self.pose_reg(base_features)#pose
        scale = self.scale_reg(base_features)
        trans = self.trans_reg(base_features)
        # rot = self.rot_reg(base_features)
        if self.ifRender:
            texture_params = self.tex_reg(base_features)#shape
        else:
            texture_params = torch.zeros(bs, self.tex_ncomp).to(device)
            
        if self.use_mean_shape:
            shape_params = torch.zeros(bs, self.shape_ncomp).to(device)
        else:
            shape_params = self.shape_reg(base_features)#shape

        return {
            'pose_params': pose_params, 
            'shape_params': shape_params, 
            'texture_params': texture_params, 
            'scale': scale, 
            'trans': trans, 
            # 'rot':rot
        }

class LightEstimator(nn.Module):
    def __init__(self, num_channel=32, dim_in=56,mode='surf'):
        super(LightEstimator, self).__init__()
        self.base_layers = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=10, stride=4, padding=1),#[48,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#[48,6,6]
            nn.Conv2d(48, 64, kernel_size=3),#[64,4,4]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#[64,2,2]
        )
        self.light_reg = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 11),
            #nn.Sigmoid()
        )
        self.mode = mode
        self.light_reg.apply(weights_init)


    def forward(self, low_features):
        base_features = self.base_layers(low_features)#[b,64,2,2]
        base_features = base_features.view(base_features.shape[0],-1)##[B,256]
        # lighting
        lights = self.light_reg(base_features)#[b,11]
        return lights


def normalize_batch_3C(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

def normalize_batch_4C(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406, 0.5]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225, 1]).view(-1, 1, 1) # heatmap will be [-0.5, 0.5]
    return (batch - mean) / std

# custom weights initialization called on Encoder
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Block') == -1 and classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1: 
        init.normal_(m.weight.data, 1.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight')  and m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Block') == -1 and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_uniform_(m.weight.data)
        init.normal_(m.weight.data, std=0.00001) 
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

class MMPool(nn.Module):
    # MMPool zhedong zheng
    def __init__(self, shape=(1,1), dim = 1, p=0., eps=1e-6):
        super(MMPool,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
        self.shape = shape

    def forward(self, x):
        return self.mmpool(x, shape = self.shape,  p=self.p, eps=self.eps)

    def mmpool(self, x, shape, p, eps):
        s = x.shape
        x_max = torch.nn.functional.adaptive_max_pool2d(x, output_size=shape)
        x_avg = torch.nn.functional.adaptive_avg_pool2d(x, output_size=shape)
        w = torch.sigmoid(self.p)
        x = x_max*w + x_avg*(1-w)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'


class Base_4C(nn.Module):
    def __init__(self, nc=4, nk=5, norm = 'bn', coordconv=True):
        super(Base_4C, self).__init__()
        # 2-4-4-3 = 12 resblocks = 24 conv
        block1 = Conv2dBlock(nc, 36, nk, stride=2, padding=nk//2, coordconv=coordconv)  #128 -> 64
        block2 = [ResBlock_half(36, norm=norm), ResBlock(72, norm=norm)] #64 -> 32
        block3 = [ResBlock_half(72, norm=norm), ResBlock(144, norm=norm), ResBlock(144, norm=norm), ResBlock(144, norm=norm)]  #32 -> 16
        block4 = [ResBlock_half(144, norm=norm), ResBlock(288, norm=norm), ResBlock(288, norm=norm), ResBlock(288, norm=norm)] #16 -> 8
        block5 = [ResBlock(288, norm=norm), ResBlock(288, norm=norm), ResBlock(288, norm=norm)] #8->8

        all_blocks = [block1, *block2, *block3] #, avgpool]
        self.layer3 = nn.Sequential(*all_blocks)
        self.layer4 = nn.Sequential(*block4)
        self.layer5 = nn.Sequential(*block5)
        # 8*8*512
        self.layer3.apply(weights_init)
        self.layer4.apply(weights_init)
        self.layer5.apply(weights_init)
    def forward(self, x):
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4) 
        return x4 + x5

class UNet_4C(nn.Module):
    def __init__(self, nc=4, nk=5, norm = 'bn', coordconv=True):
        super(UNet_4C, self).__init__()
        # 2-4-4-3 = 12 resblocks = 24 conv
        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=nk//2, coordconv=coordconv)  #128 -> 64
        block2 = [ResBlock_half(32, norm=norm), ResBlock(64, norm=norm)] #64 -> 32
        block3 = [ResBlock_half(64, norm=norm), ResBlock(128, norm=norm), ResBlock(128, norm=norm), ResBlock(128, norm=norm)]  #32 -> 16
        block4 = [ResBlock_half(128, norm=norm), ResBlock(256, norm=norm), ResBlock(256, norm=norm), ResBlock(256, norm=norm)] #16 -> 8
        block5 = [ResBlock_half(256, norm=norm), ResBlock(512, norm=norm), ResBlock(512, norm=norm)] #8->4

        all_blocks = [block1, *block2, ] #, avgpool]
        self.layer2 = nn.Sequential(*all_blocks)
        self.layer3 = nn.Sequential(*block3)
        self.layer4 = nn.Sequential(*block4)
        self.layer5 = nn.Sequential(*block5)
        # 4*2*512
        up1 = [Conv2dBlock(512, 256, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(256), nn.Upsample(scale_factor=2)]
        # 8*4*256 + 8*4*256 
        up2 = [Conv2dBlock(512, 128, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(128), nn.Upsample(scale_factor=2)]
        # 32*32*128 + 32*32*128 =  32*32*256
        up3 = [Conv2dBlock(256, 64, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(64), nn.Upsample(scale_factor=2)]
        up4 = [Conv2dBlock(128, 32, 3, 1, 1, norm='none',  activation='none', padding_mode='zeros'), ResBlock(32)]

        self.layer2.apply(weights_init)
        self.layer3.apply(weights_init)
        self.layer4.apply(weights_init)
        self.layer5.apply(weights_init)
        self.up1 = nn.Sequential(*up1)
        self.up2 = nn.Sequential(*up2)
        self.up3 = nn.Sequential(*up3)
        self.up4 = nn.Sequential(*up4)
        self.up1.apply(weights_init)
        self.up2.apply(weights_init)
        self.up3.apply(weights_init)
        self.up4.apply(weights_init)


    def forward(self, x):
        x2 = self.layer2(x) # 64 channel
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4) 
        y = self.up1(x5) # 256 channel
        y = self.up2(torch.cat((y,x4),dim=1)) # 128 channel
        y = self.up3(torch.cat((y,x3),dim=1)) # 64 channel
        y = self.up4(torch.cat((y,x2),dim=1)) # 32 channel

        return y


class Resnet_4C(nn.Module):
    def __init__(self, pretrain):
        super(Resnet_4C, self).__init__()
        if pretrain == 'res50':
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet18(pretrained=True)
        weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
        model.conv1.weight.data[:, :3] = weight
        model.conv1.weight.data[:, 3] = torch.mean(weight, dim=-1) * 0.1

        model.layer4[0].downsample[0].stride = (1,1)
        model.layer4[0].conv1.stride = (1,1)
        model.layer4[0].conv2.stride = (1,1)
        self.model = model
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x 

class HRnet_4C(nn.Module):
    def __init__(self, pretrain, if_4c=False):
        super(HRnet_4C, self).__init__()
        if pretrain == 'hr18':
            model = timm.create_model('hrnet_w18', pretrained=True, features_only=True, out_indices=[3,4])
        elif pretrain == 'hr18sv2':
            model = timm.create_model('hrnet_w18_small_v2', pretrained=True, features_only=True, out_indices=[3,4]) #  2: [b, 256, 28, 28] = 200704, 3: [b, 512, 14, 14] = 100352, 4: [b, 1024, 7, 7] = 50176
        elif pretrain == 'hr18sv1':
            model = timm.create_model('hrnet_w18_small', pretrained=True, features_only=True, out_indices=[3,4])
        if if_4c:
            # weight initialization: the weight of the 4th channel of the first layer is the mean of the weight.
            weight = model.conv1.weight.clone()
            model.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1, bias=False) #here 4 indicates 4-channel input
            model.conv1.weight.data[:, :3] = weight
            model.conv1.weight.data[:, 3] = torch.mean(weight, dim=-1)  #model.conv1.weight[:, 0]
        self.model = model
    def forward(self, x):
        low_features, features = self.model(x) # [b, 512, 14, 14] = 100352, 4: [b, 1024, 7, 7] = 50176
        return low_features, features 

class ResBlock(nn.Module):
    def __init__(self, dim, norm='bn', activation='lrelu', padding_mode='zeros', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        if norm == 'ibn':
           norm2 = 'bn'
        else: 
           norm2 = norm
        if res_type=='basic':
            model += [Conv2dBlock(dim ,dim//2, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim//2 ,dim, 3, 1, 1, norm=norm2, activation='none', padding_mode=padding_mode)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm2, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', padding_mode=padding_mode)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += 0.2 * residual # to help initial learning
        return out

class ResBlock_half(nn.Module):
    def __init__(self, dim, norm='bn', activation='lrelu', padding_mode='zeros', res_type='basic'):
        super(ResBlock_half, self).__init__()

        model = []
        if norm == 'ibn':
           norm2 = 'bn'
        else:
           norm2 = norm
        if res_type=='basic':
            model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm2, activation='none', padding_mode=padding_mode)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm2, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', padding_mode=padding_mode)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = self.model(x)
        out = torch.cat([out,residual], dim=1)
        return out

class AddCoords1d(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim)
        """
        batch_size, _, x_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).view(1, 1, x_dim)

        xx_channel = xx_channel.float() / (x_dim - 1)

        xx_channel = xx_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1)  # batchsize, 1, x_dim

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor)], dim=1)
        
        return ret


class AddCoords2d(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='lrelu', padding_mode='zeros', dilation=1, fp16 = False, coordconv = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        # initialize convolution
        self.coordconv = coordconv
        if self.coordconv:
            input_dim = input_dim + 2
            self.addcoords = AddCoords2d(with_r=False)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding, padding_mode=padding_mode, dilation=dilation, bias=self.use_bias)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ibn':
            self.norm = IBN(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16 = fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


    def forward(self, x):
        if self.coordconv:
            x = self.addcoords(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

        
class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True) # use affine.
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
