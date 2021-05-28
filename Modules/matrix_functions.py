import warnings
import numpy as np


def get_matrix_functions(boost=False, device='cpu'):
    einsum = None
    matmul = None
    torch_used = False
    if boost:
        try:
            import torch
            torch_used = True
            name = 'CPU'
            if device == 'gpu' or device == 'GPU':
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    name = 'GPU'
                else:
                    device = torch.device('cpu')
                    warnings.warn('No cuda device found.')
            print('Using Torch ' + name + ' backend.')

            def einsum(s, x1, x2):
                x1 = torch.tensor(x1).to(device)
                x2 = torch.tensor(x2).to(device)
                return np.array(torch.einsum(s, x1, x2).to('cpu'))

            def matmul(x1, x2):
                x1 = torch.tensor(x1).to(device)
                x2 = torch.tensor(x2).to(device)
                return np.array(torch.matmul(x1, x2).to('cpu'))
        except ImportError:
            torch = None
            warnings.warn('No Torch module found.')
    if not torch_used:
        matmul = np.matmul
        einsum = np.einsum
        print('Using Pure NumPy backend.')

    return matmul, einsum
