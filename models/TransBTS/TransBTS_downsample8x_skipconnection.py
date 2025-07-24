import torch
import torch.nn as nn
from models.TransBTS.Transformer import TransformerModel
from models.TransBTS.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from models.TransBTS.Unet_skipconnection import Unet


class TransformerBTS(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(TransformerBTS, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:

            self.conv_x = nn.Conv3d(
                128,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )
        
        in_channels = 1
        # in_channels = 4

        self.Unet = Unet(in_channels=in_channels, base_channels=16, num_classes=4)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)



        self.pool = nn.AdaptiveAvgPool1d(1)   # Pool over sequence length
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()           # Output between 0 and 1


    def encode(self, x):
        # print("Inside encoder")
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x1_1, x2_1, x3_1, x = self.Unet(x)
            # print("Inside encoder 1")
            x = self.bn(x)
            # print("Inside encoder 2")
            x = self.relu(x)

            # print("Inside encoder 3")
            x = self.conv_x(x)

            # print("Inside encoder 4")
            x = x.permute(0, 2, 3, 4, 1).contiguous()

            # print("Inside encoder 5")
            x = x.view(x.size(0), -1, self.embedding_dim)

            # print("Inside encoder 6")

        else:
            x = self.Unet(x)

            # print("Inside encoder 1")
            x = self.bn(x)

            # print("Inside encoder 2")
            x = self.relu(x)

            # print("Inside encoder 3")
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )

            # print("Inside encoder 4")
            x = x.view(x.size(0), x.size(1), -1, 8)

            # print("Inside encoder 5")
            x = x.permute(0, 2, 3, 1).contiguous()

            # print("Inside encoder 6")
            x = x.view(x.size(0), -1, self.flatten_dim)

            # print("Inside encoder 7")
            x = self.linear_encoding(x)
        

        # print("Inside encoder 8")

        x = self.position_encoding(x)

        # print("Inside encoder 9")
        x = self.pe_dropout(x)

        # print("Inside encoder 10")

        # apply transformer
        x, intmd_x = self.transformer(x)

        # print("Inside encoder 11")
        x = self.pre_head_ln(x)

        # print("Inside encoder 12")

        return x1_1, x2_1, x3_1, x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")
    
    def custom_decoder(self,x):
        # x shape: [1, 4096, 512] → transpose to [1, 512, 4096]
        x = x.transpose(1, 2)
        x = self.pool(x)                      # → [1, 512, 1]
        x = x.squeeze(-1)                     # → [1, 512]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)                       # → [1, 1]
        x = self.sigmoid(x)                   # [0, 1]
        return x * 100                        # scale to [0, 100]


    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):
        # print("log1")

        x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs = self.encode(x)
        # print('x1_1:- ',x1_1.shape)
        # print('x2_1:- ',x2_1.shape)
        # print('x3_1:- ',x3_1.shape)
        # print('encoder_output:- ',encoder_output.shape)
        # print('intmd_encoder_outputs:- ',intmd_encoder_outputs)


        Custom_Decoder_Output = self.custom_decoder(encoder_output)


        return Custom_Decoder_Output
















        print("log2")

        decoder_output = self.decode(
            x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )






















        print("log3")

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output
        
        print("log4")

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class BTS(TransformerBTS):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(BTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        # self.num_classes = num_classes

        # self.Softmax = nn.Softmax(dim=1)

        # self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        # self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        # self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        # self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//8)

        # self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
        # self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//16)

        # self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
        # self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//32)

        # self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)


    # def decode(self, x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):

    #     assert intmd_layers is not None, "pass the intermediate layers for MLA"
    #     encoder_outputs = {}
    #     all_keys = []
    #     for i in intmd_layers:
    #         val = str(2 * i - 1)
    #         _key = 'Z' + str(i)
    #         all_keys.append(_key)
    #         encoder_outputs[_key] = intmd_x[val]
    #     all_keys.reverse()

    #     x8 = encoder_outputs[all_keys[0]]
    #     x8 = self._reshape_output(x8)
    #     x8 = self.Enblock8_1(x8)
    #     x8 = self.Enblock8_2(x8)

    #     y4 = self.DeUp4(x8, x3_1)  # (1, 64, 32, 32, 32)
    #     y4 = self.DeBlock4(y4)

    #     y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
    #     y3 = self.DeBlock3(y3)

    #     y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
    #     y2 = self.DeBlock2(y2)

    #     y = self.endconv(y2)      # (1, 4, 128, 128, 128)
    #     y = self.Softmax(y)
    #     return y

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1




def TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned"):

    if dataset.lower() == 'brats':
        img_dim = 128
        num_classes = 4

    # num_channels = 4
    num_channels = 1

    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = BTS(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        model.cuda()
        y = model(x)
        print(y.shape)
