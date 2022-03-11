from torch.autograd import Variable
import torch.onnx
import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act2 = nn.ReLU()

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.act1(o1)
        o3 = self.conv2(o2)
        out = self.act2(o3)
        return out

def test_onnx():
    m = test()
    f = open('./test/raw_list.txt','w')
    proj = 'your path'
    input = Variable(torch.randn(1, 3, 128, 128))
    input = input.cuda()
    for i in range(100):
        input = Variable(torch.randn(1, 3, 128, 128))
        input = input.cuda()
        raw = input.permute(0, 2, 3, 1).data.cpu().numpy().astype('float32')
        raw.tofile('./test/test%d.raw'%i)
        f.write(proj+'test%d.raw'%i+'\n')
    f.close()
    if torch.cuda.is_available():
        m.cuda()
    torch.onnx.export(m,
                      (input),
                      "./test/test.onnx",
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])
    print('end')

def main():
    test_onnx()

if __name__ == '__main__':
    main()

# snpe-1.54.2.2899
# snpe-onnx-to-dlc -i test.onnx -o test.dlc
# snpe-dlc-quantize --input_dlc test.dlc --input_list raw_list.txt --output_dlc test_quantized.dlc --enable_htp
'''
[INFO] InitializeStderr: DebugLog initialized.
[INFO] Writing intermediate model
[INFO] Setting activation for layer: input and buffer: input
[INFO] bw: 8, min: -5.231276, max: 5.109619, delta: 0.040553, offset: -129.000000
[INFO] Setting activation for layer: Conv_0 and buffer: 5
[INFO] bw: 8, min: -3.248278, max: 3.273855, delta: 0.025577, offset: -127.000000
[INFO] Setting activation for layer: LeakyRelu_1 and buffer: 6
[INFO] bw: 8, min: -0.646348, max: 3.277907, delta: 0.015389, offset: -42.000000
[INFO] Setting activation for layer: Conv_2 and buffer: 7
[INFO] bw: 8, min: -1.517784, max: 1.188754, delta: 0.010614, offset: -143.000000
[INFO] Setting activation for layer: Relu_3 and buffer: output
[INFO] bw: 8, min: 0.000000, max: 1.192410, delta: 0.004676, offset: 0.000000
[INFO] Writing quantized model to: test_quantized.dlc
[DSP_TF8 : 0 1 2 3 4 ] ::1
[INFO] SNPE HTP Offline Prepare: Creating Subnet Record for layers: 0-4.
[2] QnnDsp Setting graph 256 vtcm_mb 4
[2] QnnDsp Graph Input Tensor 4 InputDef[2, 0]
[2] QnnDsp Graph Output Tensor 5 InputDef[32, 0]
[WARNING] Converting TF quantized 8->32 bit. Consider quantizing directly from float for the best accuracy.
[WARNING] Converting TF quantized 8->32 bit. Consider quantizing directly from float for the best accuracy.
[INFO] SNPE HTP Offline Prepare: Done creating QNN HTP graph cache for Vtcm size 4 MB.
[INFO] Successfully compiled HTP metadata into DLC.
[INFO] DebugLog shutting down.

'''