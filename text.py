import arch_str
import torch.onnx
import config as cfg
import torch.nn as nn
from structure import MainDenoise
from torch.autograd import Variable

#Function to Convert to ONNX
def Convert_ONNX():

    # set the model to inference mode
    model.eval()
    h = 544
    w = 960
    #########################################################################
    # ft
    h0_row = model.ft.w1
    h1_row = model.ft.w2
    h0_row_t = model.ft.w1.transpose(2, 3)
    h1_row_t = model.ft.w2.transpose(2, 3)
    h00_row = h0_row * h0_row_t
    h01_row = h0_row * h1_row_t
    h10_row = h1_row * h0_row_t
    h11_row = h1_row * h1_row_t
    filters1 = [h00_row, h01_row, h10_row, h11_row]
    zeros = torch.zeros_like(h00_row, device=h00_row.device)
    filters_g1 = []
    for i in range(4):
        tmp = filters1[i]
        g = torch.cat([torch.cat([tmp, zeros, zeros, zeros], dim=1),
                       torch.cat([zeros, tmp, zeros, zeros], dim=1),
                       torch.cat([zeros, zeros, tmp, zeros], dim=1),
                       torch.cat([zeros, zeros, zeros, tmp], dim=1)], dim=0)
        filters_g1.append(g)
    filters_ft = torch.cat(filters_g1, dim=0)  # .to('cuda')
    model.ft.w1.weight = nn.Parameter(filters_ft)
    # fti
    g0_col = model.fti.w1
    g1_col = model.fti.w2
    g0_col_t = model.fti.w1.transpose(2, 3)
    g1_col_t = model.fti.w2.transpose(2, 3)
    g00_col = g0_col * g0_col_t
    g01_col = g0_col * g1_col_t
    g10_col = g1_col * g0_col_t
    g11_col = g1_col * g1_col_t
    filters2 = [g00_col, g01_col, g10_col, g11_col]
    zeros = torch.zeros_like(g00_col, device=g00_col.device)
    filters_g2 = []
    for i in range(4):
        tmp = filters2[i]
        g = torch.cat([torch.cat([tmp, zeros, zeros, zeros], dim=1),
                       torch.cat([zeros, tmp, zeros, zeros], dim=1),
                       torch.cat([zeros, zeros, tmp, zeros], dim=1),
                       torch.cat([zeros, zeros, zeros, tmp], dim=1)], dim=0)
        filters_g2.append(g)
    filters_fti = torch.cat(filters_g2, dim=0)  # .to('cuda')
    model.fti.net.weight = nn.Parameter(filters_fti)
    #########################################################################
    model.ct0.net.weight = model.ct.net.weight
    model.ct1.net.weight = model.ct.net.weight

    model.cti_fu.net.weight = model.cti.net.weight
    model.cti_de.net.weight = model.cti.net.weight
    model.cti_re.net.weight = model.cti.net.weight

    model.ft_00.net.weight = model.ft.net.weight
    model.ft_10.net.weight = model.ft.net.weight
    model.ft_01.net.weight = model.ft.net.weight
    model.ft_11.net.weight = model.ft.net.weight
    model.ft_02.net.weight = model.ft.net.weight
    model.ft_12.net.weight = model.ft.net.weight

    model.fti_d2.net.weight = model.fti.net.weight
    model.fti_d1.net.weight = model.fti.net.weight
    model.fti_fu.net.weight = model.fti.net.weight
    model.fti_de.net.weight = model.fti.net.weight
    model.fti_re.net.weight = model.fti.net.weight
    #########################################################################
    #########################################################################
    # Let's create a dummy input tensor
    input = Variable(torch.randn(1, 8, h, w))
    input = input.cuda()
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    inputs = {
        "input": input,
    }
    # Export the model
    torch.onnx.export(model,
                      (inputs["input"],),
                      "./%s/model_qua_arch.onnx" % (cfg.model_name),
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input', ],
                      output_names=output_names
                      )
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()
    model =  arch_str.EMVD(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    checkpoint = torch.load(cfg.model_save_root)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)


    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX()
