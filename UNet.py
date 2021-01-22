import torch
import torch.nn as nn

from utils import init_weights, count_param

class multi_head_attention_2d(torch.nn.Module):
    def __init__(self, in_channel, key_filters, value_filters,
                                                        output_filters, num_heads, dropout_prob=0.5, layer_type='SAME'):
        super().__init__()
        """Multihead scaled-dot-product attention with input/output transformations.
        
        Args:
            inputs: a Tensor with shape [batch, h, w, channels]
            key_filters: an integer. Note that queries have the same number 
                of channels as keys
            value_filters: an integer
            output_depth: an integer
            num_heads: an integer dividing key_filters and value_filters
            layer_type: a string, type of this layer -- SAME, DOWN, UP
        Returns:
            A Tensor of shape [batch, _h, _w, output_filters]
        
        Raises:
            ValueError: if the key_filters or value_filters are not divisible
                by the number of attention heads.
        """

        if key_filters % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                            "attention heads (%d)." % (key_filters, num_heads))
        if value_filters % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                            "attention heads (%d)." % (value_filters, num_heads))
        if layer_type not in ['SAME', 'DOWN', 'UP']:
            raise ValueError("Layer type (%s) must be one of SAME, "
                            "DOWN, UP." % (layer_type))

        self.num_heads = num_heads
        self.layer_type = layer_type

        self.QueryTransform = None
        if layer_type == 'SAME':
            self.QueryTransform = nn.Conv2d(in_channel, key_filters, kernel_size=1, stride=1,
                                padding=0, bias=True)
        elif layer_type == 'DOWN':
            self.QueryTransform = nn.Conv2d(in_channel, key_filters, kernel_size=3, stride=2,
                                padding=1, bias=True)    # author use bias
        elif layer_type == 'UP':
            self.QueryTransform = nn.ConvTranspose2d(in_channel, key_filters, kernel_size=3, stride=2,
                                padding=1, bias=True)

        self.KeyTransform = nn.Conv2d(in_channel, key_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.ValueTransform = nn.Conv2d(in_channel, value_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.attention_dropout = nn.Dropout(dropout_prob)

        self.outputConv = nn.Conv2d(value_filters, output_filters, kernel_size=1, stride=1, padding=0, bias=True)

        self._scale = (key_filters // num_heads) ** 0.5

    def forward(self, inputs):
        """
        :param inputs: B, C, H, W
        :return: inputs: B, Co, Hq, Wq
        """

        if self.layer_type == 'SAME' or self.layer_type == 'DOWN':
            q = self.QueryTransform(inputs)
        elif self.layer_type == 'UP':
            q = self.QueryTransform(inputs, output_size=(inputs.shape[2]*2, inputs.shape[3]*2))

        # [B, Hq, Wq, Ck]
        k = self.KeyTransform(inputs).permute(0, 2, 3, 1)
        v = self.ValueTransform(inputs).permute(0, 2, 3, 1)
        q = q.permute(0, 2, 3, 1)

        Batch, Hq, Wq = q.shape[0], q.shape[1], q.shape[2]

        #[B, H, W, N, Ck]
        k = self.split_heads(k, self.num_heads)
        v = self.split_heads(v, self.num_heads)
        q = self.split_heads(q, self.num_heads)

        #[(B, H, W, N), c]
        k = torch.flatten(k, 0, 3)
        v = torch.flatten(v, 0, 3)
        q = torch.flatten(q, 0, 3)

        # normalize
        q = q / self._scale
        # attention
        #[(B, Hq, Wq, N), (B, H, W, N)]
        print(q.shape)
        print(k.shape)
        exit()
        A = torch.matmul(q, k.transpose(0, 1))
        A = torch.softmax(A, dim=1)
        A = self.attention_dropout(A)

        # [(B, Hq, Wq, N), C]
        O =  torch.matmul(A, v)
        # [B, Hq, Wq, C]
        O = O.view(Batch, Hq, Wq, v.shape[-1]*self.num_heads)
        # [B, C, Hq, Wq]
        O = O.permute(0, 3, 1, 2)
        # [B, Co, Hq, Wq]
        O = self.outputConv(O)

        return O

    def split_heads(self, x, num_heads):
        """Split channels (last dimension) into multiple heads.

        Args:
            x: a Tensor with shape [batch, h, w, channels]
            num_heads: an integer

        Returns:
            a Tensor with shape [batch, h, w, num_heads, channels / num_heads]
        """

        channel_num = x.shape[-1]
        return x.view(x.shape[0], x.shape[1], x.shape[2], num_heads, int(channel_num/num_heads))

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, dilation=1),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, dilation=1),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)



class UNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        #self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1), 
        #                                    nn.Sigmoid())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        #print(inputs.size())

        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256

        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final = self.final(up1)

        return final

class simpleUNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        #self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1), 
        #                                    nn.Sigmoid())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        #print(inputs.size())

        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256

        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final = self.final(up1)

        return final
 

'''it dont work'''
class attentionUNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(attentionUNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = multi_head_attention_2d(self.in_channels, 10, 10, filters[0], 2, 0.5, 'SAME')
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        #self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1), 
        #                                    nn.Sigmoid())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        #print(inputs.size())

        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256

        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final = self.final(up1)

        return final

       

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2,1,128,128)).cuda()
    model = UNet(in_channels = x.size()[1]).cuda()

    print(model)

    param = count_param(model)
    y = model(x)
    print('Output shape:',y.shape)
    print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
