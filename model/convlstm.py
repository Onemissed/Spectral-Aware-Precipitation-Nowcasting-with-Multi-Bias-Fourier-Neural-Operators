import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Paper: Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., Woo, W. C. (2015)
#        Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.
#        https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf
# -----------------------------------------------------------------------------

class ConvLSTM_model(nn.Module):
    def __init__(self, num_layers, num_hidden, filter_size, stride, layer_norm, args):
        super(ConvLSTM_model, self).__init__()

        self.args = args
        self.frame_channel = args.patch_size * args.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = args.img_width // args.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], width, filter_size, stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.args.total_length - 1):
            # Schedule sampling
            if t < self.args.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.args.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.args.input_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        return next_frames


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h + c_new)
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new