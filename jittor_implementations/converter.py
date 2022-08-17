map = {'torch.IntTensor': 'jt.int32',
       '.data.copy_': '=',
       '.copy': '=',
       'torch.FloatTensor': 'jt.float32',
       'torch.cat': 'jt.concat',
       '.contiguous': '',
       'squeeze()': 'squeeze(-1)',
       '.cpu()': '',
       'boxes.nelement()': 'len(boxes)',
       '.ndim': 'ndim',
       'torch.from_numpy': 'jt.array',
       'self.training()': 'self.is_train',
       '.type()': '.dtype',
       'squeeze': 'squeeze(-1)',
       'keepdim': 'keepdims',  # Word = True
       '.add': ' += ', # Word = True
       '.mul': ' *　', # Word = True
       '.cpu.data': '',
       'dim()': 'ndim',
       'torch.cuda.is_available()': 'jt.has_cuda',
       'torch.cuda.empty_cache()': 'jt.gc()'
       }