import torch
import torch.nn as nn

class Access(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Access, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden
    
model = Access(4345, 256, 2803)
mod = 'encoder_model.pth'
state_dict = torch.load(mod, map_location=torch.device('cpu'))

# Print the keys in the state_dict
for key,value in state_dict.items():
    print("Key:", key)
    print("Shape:", value.shape)
    print()

# # Load the state_dict into the model
# model.load_state_dict(state_dict)
# 
# # Display the entire state_dict
# print("\nState_dict:")
# print(model.state_dict())