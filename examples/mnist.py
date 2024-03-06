import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

ENCODE = 'HRR' # 'HRR' is the multi-level encoding or 'MAP' is the bipolar encoding
MODEL = 'Random_projection' # 'Random_projection' or 'Record_based' or 'Ngram'

CONTINUE_BASIS = False
DIMENSIONS = 10000
IMG_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 100 # for GPUs with enough memory we can process multiple images at ones
EPOCHS = 13
LEVEL = 0 # 0 full precision, 2 for 2-bits, 4 for 3-bits

transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features, vsa=ENCODE)
        self.value = embeddings.Level(levels, out_features, vsa=ENCODE)

        if CONTINUE_BASIS:
            self.value.weight.data = self.value.weight.data.round()

        if ENCODE == 'HRR' and LEVEL > 0:
            scale = 1 / torch.sqrt(torch.tensor(out_features)) / LEVEL * 2
            self.position.weight.data = (self.position.weight.data/scale).round() * scale
            self.value.weight.data = (self.value.weight.data/scale).round() * scale

    def forward(self, x):
        x = self.flatten(x)
        if MODEL == 'Random_projection':
            sample_hv = x @ self.position.weight
        elif MODEL == 'Record_based':
            sample_hv = torchhd.bind(self.position.weight, self.value(x))
            sample_hv = torchhd.multiset(sample_hv)
        else: # MODEL == 'Ngram'
            sample_hv = torchhd.bind(self.position.weight, self.value(x))
            sample_hv = torchhd.ngrams(sample_hv)
        return torchhd.hard_quantize(sample_hv)


encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)
samples_hvs = torch.zeros(60000, DIMENSIONS, device=device)
model2 = Centroid(DIMENSIONS, num_classes)
model2 = model2.to(device)
with torch.no_grad():
    batch_id = 0
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)
        samples_hvs[batch_id * BATCH_SIZE:batch_id * BATCH_SIZE + BATCH_SIZE] = encode(samples)
        model.add(samples_hvs[batch_id * BATCH_SIZE:batch_id * BATCH_SIZE + BATCH_SIZE], labels)
        batch_id += 1

model2.weight.data = model.weight.clone()

if True:
    with torch.no_grad():
        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        for i in range(EPOCHS):
            model.weight.data = model2.weight.clone()
            model.normalize()
            batch_id = 0
            for samples, labels in train_ld:
                samples = samples.to(device)
                labels = labels.to(device)
                output = model(samples_hvs[batch_id * BATCH_SIZE:batch_id * BATCH_SIZE + BATCH_SIZE], dot=True)
                accuracy.update(output.cpu(), labels.cpu())
                for j in range(BATCH_SIZE):
                    if torch.argmax(output[j]) != labels[j]:
                        model2.add(samples_hvs[batch_id * BATCH_SIZE + j].view(1,-1), labels[j])
                        model2.add(torchhd.negative(samples_hvs[batch_id * BATCH_SIZE + j].view(1,-1)), torch.argmax(output[j]))
                batch_id += 1
            print('Epoch',i,f"Training accuracy of {(accuracy.compute().item() * 100):.3f}%")

            if i >= 10:
                accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

                with torch.no_grad():
                    model.normalize()

                    for samples, labels in tqdm(test_ld, desc="Testing"):
                        samples = samples.to(device)

                        samples_hv = encode(samples)
                        outputs = model(torch.tensor(samples_hv), dot=True)
                        accuracy.update(outputs.cpu(), labels)

                print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
