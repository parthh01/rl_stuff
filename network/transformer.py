# vanilla transformer as a base class to use for DQN experiments

import torch 
import torch.nn as nn

class CustomEncoder(nn.Transformer):
	"""Input will be of size (N,m,P) where N is sample size, m is feature size, p is sequence length"""
	def __init__(self,sequence_length,num_heads,num_layers=1):
		super().__init__()
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=sequence_length,nhead=num_heads,batch_first=True)
		self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=num_layers)




if __name__ == "__main__":
	x = torch.rand(10,32,512)
	mask = torch.triu(torch.ones(32,32)*float('-inf'),diagonal=1)
	enc = CustomEncoder(x.shape[-1],8,6)
	out = enc.encoder(x,mask=mask) #key_padding_mask if mask needs to be different per element in the batch 
	print(out.shape)
