import torch
import torch.nn as nn                           # 신경망 구성 도구
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms    # 숫자 빅데이터 파일
from PIL import Image                           # 이미지 인식
import os

########################################
# 1. 기본 설정
########################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

########################################
# 2. MNIST 데이터셋 로드 (0~9 전체)
########################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

########################################
# 3. CNN 모델 정의 (0~9 → 10 클래스)
########################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 클래스 (0~9)

    def forward(self, x):
        x = torch.relu(self.conv1(x))    # (B,32,28,28)
        x = self.pool(x)                 # (B,32,14,14)
        x = torch.relu(self.conv2(x))    # (B,64,14,14)
        x = self.pool(x)                 # (B,64,7,7)
        x = x.view(x.size(0), -1)        # 평탄화
        x = torch.relu(self.fc1(x))      # (B,128)
        x = self.fc2(x)                  # (B,10)
        return x

model = SimpleCNN().to(DEVICE)

########################################
# 4. 손실함수 & 최적화
########################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

########################################
# 5. 학습 함수
########################################
def train():
    model.train()  

    for epoch in range(1, EPOCHS + 1):

        running_loss = 0.0    # 이번 에폭 동안 발생한 손실 누적값
        correct = 0           # 맞게 예측한 이미지 개수
        total = 0             # 전체 예측한 이미지 개수

        for images, labels in train_loader:
            # DataLoader가 배치 단위로 (이미지, 정답) 데이터를 제공합니다.
            # 예: batch_size=64 → 한 번에 64장씩 들어옴

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            # 이전 함수에서 계산된 gradient(기울기)를 초기화

            outputs = model(images)
            # 모델에 이미지를 넣어 forward 연산을 수행하고, 0~9 logits 출력
            # outputs 크기 예: (64, 10)

            loss = criterion(outputs, labels)
            # 손실함수를 이용해 예측값(outputs)과 정답(labels)의 차이를 계산합니다.
            # CrossEntropyLoss는 정답 클래스의 확률이 낮을수록 큰 손실을 부여합니다.

            loss.backward()
            # 역전파(Backpropagation)를 실행합니다.
            # 손실이 각 가중치에 얼마나 영향을 받았는지 기울기를 계산합니다.

            optimizer.step()
            # 계산된 기울기를 바탕으로 가중치를 업데이트합니다.
            # Loss가 더 작아지는 방향으로 weight 조정

            running_loss += loss.item() * images.size(0)
            # 현재 배치의 손실을 전체 손실에 더함
            # loss.item()은 파이썬 실수로 변환
            # images.size(0)는 배치 크기 → 평균 계산을 위해 곱해줌

            _, predicted = torch.max(outputs, 1)
            # outputs에서 가장 높은 점수를 가진 클래스를 예측 결과로 선택

            total += labels.size(0)
            # 이번 배치의 전체 이미지 수 만큼 누적

            correct += (predicted == labels).sum().item()
            # 예측(predicted)과 정답(labels)이 동일한 개수만큼 누적

        epoch_loss = running_loss / len(train_loader.dataset)
        # 에폭 전체 평균 손실 계산
        # running_loss를 전체 데이터 수로 나눔

        epoch_acc = correct / total * 100
        # 에폭 정확도 계산 (0~100%)

        print(f"[Epoch {epoch}/{EPOCHS}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        # 에폭의 평균 손실값 및 정확도를 출력하여 학습 상태를 확인합니다.


########################################
# 6. 테스트 함수
########################################
def test():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total * 100
    print(f"[Test] Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

########################################
# 7. 사용자 이미지 예측 함수
########################################
def predict_image(image_path, true_label=None):
    """
    image_path: 예측할 이미지 경로 (숫자 0~9)
    true_label: 0~9 정답(선택). 넣으면 손실값 출력.
    """
    if not os.path.exists(image_path):
        print(f"파일을 찾을 수 없습니다: {image_path}")
        return

    model.eval()

    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))

    # 일반적으로 흰 배경 + 검은 숫자면 MNIST와 반대라서 반전
    img = transforms.functional.invert(img)

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        predicted_digit = predicted.item()
        predicted_prob = probs[0, predicted_digit].item()

    print(f"\n이미지: {image_path}")
    print(f"예측 숫자: {predicted_digit}  (확률: {predicted_prob*100:.2f}%)")

    if true_label is not None:
        label_tensor = torch.tensor([true_label], device=DEVICE)
        loss = criterion(outputs, label_tensor)
        print(f"손실값 (CrossEntropyLoss): {loss.item():.4f}")

########################################
# 8. 실행부 (터미널에서 파일 이름 입력)
########################################
if __name__ == "__main__":
    print("=== MNIST 0~9 숫자 CNN 학습 시작 ===")
    train()

    print("\n=== 테스트 성능 측정 ===")
    test()

    # 테스트셋 첫 이미지로 예시 한 번
    sample_img, sample_label = test_dataset[0]
    sample_path = "sample_mnist.png"
    transforms.ToPILImage()(sample_img).save(sample_path)

    print("\n=== 테스트 샘플 예측 ===")
    predict_image(sample_path, true_label=sample_label)

    # 여기부터: 사용자 입력으로 파일명 받아서 예측
    while True:
        print("\n예측할 이미지 파일 이름을 입력하세요.")
        print("(예: my_digit.png, 그냥 엔터 치면 종료)")
        user_path = input("파일 이름: ").strip()

        if user_path == "":
            print("종료합니다.")
            break


        predict_image(user_path, true_label=true_label)

