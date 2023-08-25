import torch

print('PyTorch version is: ', torch.__version__)

if torch.cuda.is_available():
	device = torch.device('cuda')
	gpu_name = torch.cuda.get_device_name(0)
	print(f'GPU: {gpu_name}')
else:
	device = torch.device('cpu')
	print('No GPU, using CPU')

print('Available device name:', device)
