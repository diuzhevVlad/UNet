# import the necessary packages
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
		mask = PIL.Image.open(self.masks[idx])

		if self.transforms is not None:
		    image = self.transforms(image)
		    mask = self.transforms(mask)
		mask = torch.cat([(mask==0).type(torch.FloatTensor),(mask!=0).type(torch.FloatTensor)],0)
		    
		return (image, mask)