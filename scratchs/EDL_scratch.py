import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the evidence functions
def relu_evidence(logits):
    return torch.relu(logits)


def exp_evidence(logits):
    return torch.exp(torch.clamp(logits, min=-10, max=10))


def softplus_evidence(logits):
    return torch.nn.functional.softplus(logits)


# Define the KL divergence function
def KL(alpha):
    K = alpha.shape[1]
    beta = torch.ones((alpha.shape[0], K), dtype=torch.float32)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


# Define the MSE loss function
def mse_loss(labels, alpha, global_step, annealing_step):
    S = alpha.sum(dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S

    # Convert labels to one-hot encoding
    p = torch.nn.functional.one_hot(labels, num_classes=alpha.shape[1]).float()

    A = torch.sum((p - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    annealing_coef = min(1.0, global_step / annealing_step)

    alp = E * (1 - p) + 1
    C = annealing_coef * KL(alp)

    # Ensure that the loss is a scalar
    total_loss = (A + B) + C
    return total_loss.mean()  # Use .mean() to ensure the result is a scalar


# Define the LeNet model with Evidence Deep Learning (EDL)
class LeNet_EDL(nn.Module):
    def __init__(self, logits2evidence=relu_evidence, loss_function=mse_loss, lmb=0.005):
        super(LeNet_EDL, self).__init__()
        self.logits2evidence = logits2evidence
        self.loss_function = loss_function
        self.lmb = lmb
        self.global_step = 0
        self.annealing_step = 10 * 1000  # Default value, you can adjust

        # Define LeNet layers
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        print('x.shape before view: ', x.shape)
        x = x.view(-1, 50 * 4 * 4)
        print('x.shape after view: ', x.shape)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def calculate_loss(self, logits, labels):
        evidence = self.logits2evidence(logits)
        alpha = evidence + 1
        p = labels
        global_step = self.global_step
        annealing_step = self.annealing_step
        loss = self.loss_function(p, alpha, global_step, annealing_step)
        l2_loss = (torch.norm(self.fc1.weight, p=2) + torch.norm(self.fc2.weight, p=2)) * self.lmb
        return loss + l2_loss

    def get_metrics(self, logits, labels, K=10):
        evidence = self.logits2evidence(logits)  # 对应原文的e_k
        alpha = evidence + 1  # 对应原文的Dirichlet distribution parameter a_k

        # Compute uncertainty
        u = K / alpha.sum(dim=1, keepdim=True)  # 对应原文的u, uncertainty

        # Compute probabilities
        prob = alpha / alpha.sum(dim=1, keepdim=True)  # 对应原文的expected probability p_k

        # Compute mean evidence
        total_evidence = evidence.sum(dim=1, keepdim=True)
        mean_ev = total_evidence.mean().item()

        # Compute mean evidence for successful and failed predictions
        pred = logits.argmax(dim=1)
        match = (pred == labels).float()
        mean_ev_succ = (total_evidence * match).sum() / (match.sum() + 1e-20)
        mean_ev_fail = (total_evidence * (1 - match)).sum() / ((1 - match).sum() + 1e-20)

        return u, prob, mean_ev, mean_ev_succ.item(), mean_ev_fail.item()


def train_and_evaluate(model, train_loader, test_loader, epochs=50, batch_size=1000):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = model.calculate_loss(output, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}')

        model.eval()
        train_acc = 0
        test_acc = 0

        # Metrics
        train_metrics = {
            'mean_ev': 0,
            'mean_ev_succ': 0,
            'mean_ev_fail': 0
        }
        test_metrics = {
            'mean_ev': 0,
            'mean_ev_succ': 0,
            'mean_ev_fail': 0
        }

        with torch.no_grad():
            for data, target in train_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                train_acc += pred.eq(target).sum().item()
                u, prob, mean_ev, mean_ev_succ, mean_ev_fail = model.get_metrics(output, target)
                train_metrics['mean_ev'] += mean_ev
                train_metrics['mean_ev_succ'] += mean_ev_succ
                train_metrics['mean_ev_fail'] += mean_ev_fail

            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                test_acc += pred.eq(target).sum().item()
                u, prob, mean_ev, mean_ev_succ, mean_ev_fail = model.get_metrics(output, target)
                test_metrics['mean_ev'] += mean_ev
                test_metrics['mean_ev_succ'] += mean_ev_succ
                test_metrics['mean_ev_fail'] += mean_ev_fail

        train_acc /= len(train_loader.dataset)
        test_acc /= len(test_loader.dataset)

        # Average metrics
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        test_metrics = {k: v / len(test_loader) for k, v in test_metrics.items()}

        print(f'Epoch {epoch + 1}, Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}')
        print(f'Training Metrics - Mean Evidence: {train_metrics["mean_ev"]:.4f}, '
              f'Mean Evidence (Succ): {train_metrics["mean_ev_succ"]:.4f}, '
              f'Mean Evidence (Fail): {train_metrics["mean_ev_fail"]:.4f}')
        print(f'Testing Metrics - Mean Evidence: {test_metrics["mean_ev"]:.4f}, '
              f'Mean Evidence (Succ): {test_metrics["mean_ev_succ"]:.4f}, '
              f'Mean Evidence (Fail): {test_metrics["mean_ev_fail"]:.4f}')


if __name__ == '__main__':
    # Load FashionMNIST dataset from local directory
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = LeNet_EDL()
    train_and_evaluate(model, train_loader, test_loader)
