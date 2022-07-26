import torch.nn as nn

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
     
class ActivationLayer(nn.Module):

    def __init__(self, activation='lk_relu', alpha_relu=0.15, inplace=False):
        super().__init__()

        if activation =='lk_relu':
            self.activation = nn.LeakyReLU(alpha_relu)

        elif activation =='relu':
            self.activation = nn.ReLU(inplace)

        elif activation =='softmax':
            self.activation = nn.Softmax()

        elif activation =='sigmoid':
            self.activation = nn.Sigmoid()

        elif activation =='tanh':
            self.activation = nn.Tanh()

        elif activation =='selu':
            self.activation = nn.Selu()

        else :
            # Identity function
            self.activation = None

    def forward(self, x):

        if self.activation is None :
            # Identity function
            return x
        
        else :
            return self.activation(x)

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class NormalizationLayer(nn.Module):

    def __init__(self, in_features, norm_type='IN2D'):
        super().__init__()

        if norm_type.upper() == 'BN2D' :
            self.norm = nn.BatchNorm2d(in_features)

        elif norm_type.upper() == 'IN2D' :
            self.norm = nn.InstanceNorm2d(in_features)

        elif norm_type.upper() == 'BN1D' :
            self.norm = nn.BatchNorm1d(in_features)

        elif norm_type.upper() == 'IN1D' :
            self.norm = nn.InstanceNorm1d(in_features)

        elif norm_type.upper() == 'ADAIN':
            self.norm = AdaptiveInstanceNorm(in_features)

        elif norm_type.upper() == 'NONE' :
            self.norm = lambda x : x * 1.0

        else:
            raise NotImplementedError('[INFO] The Normalization layer %s is not implemented !' % norm_type)

    def forward(self, x):
        out = self.norm(x)
        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#.


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_features)

    def calc_mean_std_4D(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, input, style):
        style_mean, style_std = self.calc_mean_std_4D(style)
        out = self.norm(input)
        size = input.size()
        out = style_std.expand(size) * out + style_mean.expand(size)
        return 

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class Conv2DLayer(nn.Module):
    def __init__(   self, in_features, out_features, scale='none', norm_type='in2d', norm_before=True, activation='lk_relu', alpha_relu=0.15, 
                    interpolation_mode='bicubic', inplace=False, scale_factor=2, is_debug=False, **kwargs):
        
        super().__init__()
        
        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before
        self.scale_factor = scale_factor
        self.is_debug = is_debug
        self.norm_type = norm_type
        
        # upsampling or downsampling
        stride = 2 if scale == 'down' else 1

        if scale == 'up':
            self.scale_layer = lambda x : nn.functional.interpolate(x, scale_factor=scale_factor, mode=interpolation_mode)
        else :
            self.scale_layer = lambda x : x

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, stride=stride, **kwargs)

        # Activation layer
        self.activation = ActivationLayer(activation=activation, alpha_relu=alpha_relu, inplace=inplace)

        # Normalization layer
        self.norm = NormalizationLayer(in_features=out_features, norm_type=norm_type)

    def forward(self, x):
        
        # upsampling or downsampling 
        out = self.scale_layer(x)

        out = self.conv(out)

        if self.norm_before :
            if self.norm_type.upper() == 'ADAIN':
                out = self.norm(out, out)
            else:
                out = self.norm(out)

            out = self.activation(out)

        else :
            out = self.activation(out)
            if self.norm_type.upper() == 'ADAIN':
                out = self.norm(out, out)
            else:
                out = self.norm(out)
        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#