import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

def get_encoding(change_encoding):
    og = np.array([0, 1, 2, 3])
    encoding = {'A': 0, 'C': 1, 'U': 2, 'G': 3}
    new_encoding = {}
    if change_encoding > 3:
        np.random.seed(change_encoding)
        remapped = np.random.permutation(4) 
    else:
        remapped = (og + change_encoding) % 4

    index_to_base = {v: k for k, v in encoding.items()}
    new_encoding = {base: remapped[idx] for idx, base in index_to_base.items()}
    return new_encoding


rna_encoding = {'A': 1, 'C': 2, 'U': 3, 'G': 0}
def one_hot_encode(seq, vocab=rna_encoding):
    arr = np.array([vocab[base] for base in seq])
    return np.eye(len(vocab))[arr]  

def qary_to_strings(qary_array, encoding):
    index_to_base = {v: k for k, v in encoding.items()}
    strings = ["".join(index_to_base[val] for val in row) for row in qary_array]
    return strings

def infer(qary_array, model, device='cuda', change_encoding=0):
    """
    Inference from 2D q-ary array (originally A=0, C=1, U=2, G=3)
    using a custom encoding
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Remap using new encoding
    og = np.array([0, 1, 2, 3])
    encoding = {'A': 0, 'C': 1, 'U': 2, 'G': 3}
    new_encoding = {}
    if change_encoding > 3:
        np.random.seed(change_encoding)
        remapped = np.random.permutation(4) 
    else:
        remapped = (og + change_encoding) % 4

    index_to_base = {v: k for k, v in encoding.items()}
    new_encoding = {base: remapped[idx] for idx, base in index_to_base.items()}
    # Convert to nucleotide strings
    seqs = qary_to_strings(qary_array, new_encoding)

    # Convert strings to one-hot encoding
    one_hot_seqs = np.array([one_hot_encode(seq).flatten() for seq in seqs])
    X_tensor = torch.tensor(one_hot_seqs, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor).squeeze().cpu().numpy()

    return preds


def encode_sequences(seqs, encoding_dict):
    return np.array([[encoding_dict[char] for char in seq] for seq in seqs], dtype=int)
