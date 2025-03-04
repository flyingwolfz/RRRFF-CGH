import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
        )
    def forward(self, x):
        out1 = self.net1(x)
        return out1


class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
        )
    def forward(self, x):
        out1 = self.net1(x)
        return out1



class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 last_layer=False):
        super().__init__()
        if last_layer == True:
            self.net1 = nn.Sequential(
                #nn.Conv2d(in_channels, 4 * out_channels, 3, stride=1, padding=1),
                #nn.PixelShuffle(2),
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                #nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2, padding=2, output_padding=1),
            )
        else:
            self.net1 = nn.Sequential(
                #nn.Conv2d(in_channels, 4 * out_channels, 3, stride=1, padding=1),
                #nn.PixelShuffle(2),
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                #nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2, padding=2,output_padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                #nn.ReLU(),
            )

    def forward(self, x):
        out1 = self.net1(x)
        return out1

class DownA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.AngConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.AngConv(x)
        return out1

class DownSA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.AngConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
        )
        self.SpaConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
        )
    def forward(self, x):
        out1 = self.AngConv(x)
        out2 = self.SpaConv(x)
        out3 = torch.cat((out1, out2), -3)
        return out3

class DownS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.SpaConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels , kernel_size=3, stride=1, dilation=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
            nn.Conv2d(out_channels, out_channels , kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(),
        )
    def forward(self, x):
        out2 = self.SpaConv(x)
        return out2


class DownS_2v(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.SpaConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=(1, 2), padding=(1, 2)),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(),
        )
    def forward(self, x):
        out2 = self.SpaConv(x)
        return out2
class DownSA_2v(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.AngConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1,2), stride=(1,2), padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(),
        )
        self.SpaConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, dilation=(1,2), padding=(1,2)),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(1,2), stride=(1,2), padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(),
        )
    def forward(self, x):
        out1 = self.AngConv(x)
        out2 = self.SpaConv(x)
        out3 = torch.cat((out1, out2), -3)
        return out3
class Net(nn.Module):
    def __init__(self, input_type, channel_num):
        super().__init__()
        self.channel_num=channel_num
        self.input_type=input_type
        first_channel = 32
        if self.input_type == 'macpi':
            self.first_layer = DownS(self.channel_num, first_channel)
        elif self.input_type == '2view':
            self.first_layer = Conv(self.channel_num * 2, first_channel)
        elif self.input_type == '4view':
            self.first_layer = Conv(self.channel_num * 4, first_channel)
        elif self.input_type == '1view':
            self.first_layer = Conv(self.channel_num, first_channel)
        elif self.input_type == 'rgbd':
            self.first_layer = Conv(self.channel_num + 1, first_channel)
        elif self.input_type == 'macpi_2view':
            self.first_layer = DownS_2v(self.channel_num, first_channel)
        elif self.input_type == 'macpi_dual':
            self.first_layer = DownSA(self.channel_num*2, first_channel)
        elif self.input_type == 'macpia':
            self.first_layer = DownA(self.channel_num * 2, first_channel)
        elif self.input_type == 'macpisa':
            self.first_layer = DownSA(self.channel_num * 2, first_channel)
        elif self.input_type == 'macpid':
            self.first_layer = Down(self.channel_num * 2, first_channel)

        k = 64
        self.netdown2 = Down(first_channel, k)
        self.netdown3 = Down(k, k * 2)
        self.netdown4 = Down(k * 2, k * 4)
        self.netdown5 = Down(k * 4, k * 8)
        self.netup1 = Up(k * 8, k * 4)
        self.netup2 = Up(k * 4, k * 2)
        self.netup3 = Up(k * 2, k)
        self.netup4 = Up(k, self.channel_num*2, last_layer=True)

    def forward(self, x):

        dout1 = self.first_layer(x)

        dout2 = self.netdown2(dout1)
        dout3 = self.netdown3(dout2)
        dout4 = self.netdown4(dout3)
        dout5 = self.netdown5(dout4)

        uout1 = self.netup1(dout5)
        uout2 = self.netup2(uout1+dout4 )
        uout3 = self.netup3(uout2 + dout3)
        uout4 = self.netup4(uout3+dout2)
        if self.channel_num==1:
            out = torch.atan2(uout4[:, 0, :, :], uout4[:, 1, :, :]).unsqueeze(0) #single_color
        elif self.channel_num==3:
            out0 = torch.atan2(uout4[:, 0, :, :], uout4[:, 1, :, :]).unsqueeze(0)
            out1 = torch.atan2(uout4[:, 2, :, :], uout4[:, 3, :, :]).unsqueeze(0)
            out2 = torch.atan2(uout4[:, 4, :, :], uout4[:, 5, :, :]).unsqueeze(0)
            out=torch.cat((out0,out1,out2),1)  # bgr full color
        return out

class Net_random(nn.Module):
    def __init__(self, input_type, channel_num):
        super().__init__()
        self.channel_num=channel_num
        self.input_type=input_type
        first_channel = 32
        if self.input_type == 'macpi':
            self.first_layer = DownS(self.channel_num, first_channel-2)
        elif self.input_type == '2view':
            self.first_layer = Conv(self.channel_num * 2, first_channel)
        elif self.input_type == '4view':
            self.first_layer = Conv(self.channel_num * 4, first_channel)
        elif self.input_type == '1view':
            self.first_layer = Conv(self.channel_num, first_channel-2)
        elif self.input_type == 'rgbd':
            self.first_layer = Conv(self.channel_num + 1, first_channel-2)
        elif self.input_type == 'macpi_2view':
            self.first_layer = DownS_2v(self.channel_num, first_channel-2)
        elif self.input_type == 'macpi_dual':
            self.first_layer = DownSA(self.channel_num*2, first_channel)
        elif self.input_type == 'macpia':
            self.first_layer = DownA(self.channel_num * 2, first_channel)
        elif self.input_type == 'macpisa':
            self.first_layer = DownSA(self.channel_num * 2, first_channel)
        elif self.input_type == 'macpid':
            self.first_layer = Down(self.channel_num * 2, first_channel)

        k = 64
        self.netdown2 = Down(first_channel, k)
        self.netdown3 = Down(k, k * 2)
        self.netdown4 = Down(k * 2, k * 4)
        self.netdown5 = Down(k * 4, k * 8)
        self.netup1 = Up(k * 8, k * 4)
        self.netup2 = Up(k * 4, k * 2)
        self.netup3 = Up(k * 2, k)
        self.netup4 = Up(k, self.channel_num*2, last_layer=True)

    def forward(self, x,posx,posy):

        dout1 = self.first_layer(x)
        dout1=torch.cat((dout1,posx,posy),-3)
        dout2 = self.netdown2(dout1)
        dout3 = self.netdown3(dout2)
        dout4 = self.netdown4(dout3)
        dout5 = self.netdown5(dout4)

        uout1 = self.netup1(dout5)
        uout2 = self.netup2(uout1+dout4 )
        uout3 = self.netup3(uout2 + dout3)
        uout4 = self.netup4(uout3+dout2)
        if self.channel_num==1:
            out = torch.atan2(uout4[:, 0, :, :], uout4[:, 1, :, :]).unsqueeze(0) #single_color
        elif self.channel_num==3:
            out0 = torch.atan2(uout4[:, 0, :, :], uout4[:, 1, :, :]).unsqueeze(0)
            out1 = torch.atan2(uout4[:, 2, :, :], uout4[:, 3, :, :]).unsqueeze(0)
            out2 = torch.atan2(uout4[:, 4, :, :], uout4[:, 5, :, :]).unsqueeze(0)
            out=torch.cat((out0,out1,out2),1)  # bgr full color
        return out




if __name__ == "__main__":
    #from thop import profile
    import tools.tools as tools

    import_onnx = 1
    import_name = 'trained_model/onnx.onnx'
    load_trained = 1
    load_pth_name = 'trained_model/input_macpi_sup_focus_addsup_focus_channel_3.pth'
    show_onnx=0
    check_onnx=0
    run_num = 100

    input_type = 'macpi'

    W = 3840
    H = 2160
    channel_num=3   #1 or 3

    camera_positionx = -4.8
    camera_positiony = 6.4
    posx = torch.full((1, 1, H, W), camera_positionx / 8.0).cuda(0)
    posy = torch.full((1, 1, H, W), camera_positiony / 8.0).cuda(0)

    #net = Net(input_type=input_type, channel_num=channel_num).cuda()
    net = Net_random(input_type=input_type, channel_num=channel_num).cuda()
    net_input = tools.get_rand_input(input_type=input_type, W=W, H=H,channel_num=channel_num)

    if load_trained==1:
        net.load_state_dict(torch.load(load_pth_name))

    # flops, params = profile(net, inputs= net_input.unsqueeze(0))
    # print('   Number of parameters: %.2fM' % (params / 1e6))
    # print('   Number of FLOPs: %.2fG' % (flops * 2 / 1e9))
    out = net(net_input, posx, posy)
    #out = net(net_input)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i in range(run_num):
            out = net(net_input, posx, posy)
            #out = net(net_input)
    torch.cuda.synchronize()
    end.record()
    print('time: (ms):', start.elapsed_time(end) / run_num)

    if import_onnx == 1:
        import torch.onnx
        torch.onnx.export(net,  # model being run
                          (net_input, posx, posy),  # model input
                          import_name,  # where to save the model
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=17,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          # verbose=True,
                          )
        print('import onnx,name:', import_name)
        # torch.onnx.export(net, net_input, 'trained_model/torchonnx.onnx', input_names=['input'],
        #                   output_names=['output'])  #same
    if check_onnx==1:
        import onnx
        onnx.checker.check_model(onnx.load(import_name))
    if show_onnx==1:
        import netron
        netron.start(import_name)