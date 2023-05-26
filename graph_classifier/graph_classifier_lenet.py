from enum import Enum
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image


class GraphType(Enum):
    DOT = "dot"
    LINE = "line"
    SCATTER = "scatter"
    HORIZONTAL_BAR = "horizontal_bar"
    VERTICAL_BAR = "vertical_bar"


class GraphClassifierLenet:
    def __init__(self, model_path):
        self.model = LeNet()
        self.model.load_state_dict(torch.load(model_path))
        self.transform = transforms.Compose([
            transforms.Resize((267, 466)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def classify(self, graph_path: str) -> GraphType:
        try:
            # 加载要预测的图片并进行数据预处理
            image = Image.open(graph_path)
            image = self.transform(image)
            # 对图片进行预测
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))  # 加一维，batch size为1
                _, predicted = torch.max(output.data, 1)
                item = predicted.item()
                if item == 0:
                    return GraphType.DOT
                elif item == 1:
                    return GraphType.HORIZONTAL_BAR
                elif item == 2:
                    return GraphType.LINE
                elif item == 3:
                    return GraphType.SCATTER
                else:
                    return GraphType.VERTICAL_BAR
        except Exception:
            return GraphType.VERTICAL_BAR


# 定义LeNet网络
class LeNet(nn.Module):
    def __init__(self, num_classes=5):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 63 * 113, 120),
            # nn.Linear(16 * 109 * 59, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train():
    # 定义数据增强和标准化处理
    transform = transforms.Compose([
        transforms.Resize((267, 466)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    dataset = ImageFolder(root='dataset/train/classify', transform=transform)
    # dataset = Subset(dataset, range(100))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 定义模型和优化器
    model = LeNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (data, label) in progress_bar:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, label)
            loss.backward()
            optimizer.step()

            # 更新进度条的描述信息
            progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 在测试集上测试模型性能
        model.eval()
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (data, label) in progress_bar:
                data = data.to(device)
                label = label.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            accuracy = 100 * correct / total
            print('\nEpoch: %d, Test Accuracy: %.2f%%' % (epoch + 1, accuracy))
            torch.save(model.state_dict(), f'graph_classifier/graph_classifier_epoch{epoch + 1}.pth')


def predict():
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('graph_classifier/graph_classifier.pth'))

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((267, 466)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载要预测的图片并进行数据预处理
    image = Image.open('dataset/test/images/01b45b831589.jpg')
    image = transform(image)

    # 对图片进行预测
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # 加一维，batch size为1
        _, predicted = torch.max(output.data, 1)
        print(predicted.item())  # 输出预测的分类标签


if __name__ == '__main__':
    # predict()
    train()  # Epoch: 5, Test Accuracy: 99.22%
