import torch
import torchvision
import os

# load the saved binary program.
binary_program = torch.load("binary_program.pth", weights_only=True)
is_cuda = torch.cuda.is_available()

# same functions from main.py
def encode_input(x):
    x_expanded = torch.zeros(784, 256, device=x.device)
    for i in range(784):
        x_expanded[i, int(x[i] * 255)] = 1
    return x_expanded

class ProgramState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('rules', torch.zeros((784, 256, 10)))
    
    def update_rules(self, new_rules):
        self.rules = new_rules
    
    def compute(self, x_expanded):
        active_pixels = (x_expanded.unsqueeze(-1)) * self.rules
        return active_pixels.sum(dim=(0, 1))

download = True
if os.path.exists("MNIST"):
    download = False
test_data = torchvision.datasets.MNIST(root="./", download=download, train=False, transform=torchvision.transforms.ToTensor())

# Initialize ProgramState and load the binary program.
program_state = ProgramState()
if is_cuda:
    program_state.to("cuda")
program_state.update_rules(binary_program)

# test the binary program.
correct_predictions = 0
total_predictions = 0

for i in range(len(test_data)):
    img, label = test_data[i]
    X = img.view(-1)
    if is_cuda:
        X = X.to("cuda")
    X_encoded = encode_input(X)
    
    output = program_state.compute(X_encoded)  # compute output using program.
    soft_output = torch.nn.functional.softmax(output, dim=-1)
    pred = torch.argmax(soft_output).cpu().item()  # get predicted label.
    
    correct_predictions += (pred == label)
    total_predictions += 1

test_accuracy = (correct_predictions / total_predictions) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
