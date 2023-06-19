import torch
from torch.utils.data import Dataset
import PIL.Image

class SegmentDataset(Dataset):
	def __init__(self, images, masks, transforms):
		self.transforms = transforms
		self.images = images
		self.masks = masks

	def __len__(self):
		return len(self.images)
	

	def __getitem__(self, idx):
		image = PIL.Image.open(self.images[idx]).convert("RGB")
		if self.masks is not None:
			mask = PIL.Image.open(self.masks[idx])
		else:
			mask = None

		if self.transforms is not None:
			image = self.transforms(image)
			if mask is not None:
				mask = self.transforms(mask)
		
		if mask is not None:
			mask = torch.cat([(mask==0).type(torch.FloatTensor),(mask!=0).type(torch.FloatTensor)],0)
			
		return (image, mask)