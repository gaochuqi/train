
import arch_str
import torch.onnx
import config as cfg
form structure import MainDenoise

#Function to Convert to ONNX
def Convert_ONNX():

    # set the model to inference mode
    model.eval()
    h = 544
    w = 960
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
    model = MainDenoise()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    PATH= cfg.model_save_root
    # Let's load the model we just created and test the accuracy per label
    model.load_state_dict(torch.load(PATH))


    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX()
