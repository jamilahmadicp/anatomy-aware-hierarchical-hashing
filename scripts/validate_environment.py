#!/usr/bin/env python
import sys

def main():
    import torch, torchvision, timm, numpy, pandas, sklearn, scipy, PIL, matplotlib, yaml
    print('Python:', sys.version.replace('\n',' '))
    print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
    print('torchvision:', torchvision.__version__)
    print('timm:', timm.__version__)
    print('numpy:', numpy.__version__, 'pandas:', pandas.__version__, 'sklearn:', sklearn.__version__, 'scipy:', scipy.__version__)
    for name in ['convnext_tiny.fb_in1k','swin_tiny_patch4_window7_224.ms_in1k']:
        assert name in timm.list_models(), f'{name} not found in this timm installation'
        print('model available:', name)
    print('Environment validation PASS')
if __name__=='__main__': main()
