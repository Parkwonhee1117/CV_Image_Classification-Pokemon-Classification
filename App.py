import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

# 클래스 이름 로드 함수
def load_classes(file_path="classes.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        # classes.txt가 없는 경우 기본 Class 이름 생성
        return [f"Class_{i}" for i in range(150)]

# 모델 구조 생성 함수
def build_model(model_name='resnet50', num_classes=150):
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("지원하지 않는 모델입니다.")
    return model

# ---------------------------
# Streamlit UI 구성
# ---------------------------
def main():
    st.title("Pokemon Classifier 🦖")
    st.write("이미지를 업로드하면 포켓몬의 이름을 예측합니다.")

    classes = load_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성 및 가중치 로드
    model = build_model(model_name='resnet50', num_classes=len(classes))
    
    # 사용할 모델 가중치 파일 지정 ('exp2.pth' 또는 'best_model.pth')
    model_path = "exp2.pth" 
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.success(f"성공적으로 {model_path} 가중치를 불러왔습니다!")
    else:
        st.warning(f"⚠️ {model_path} 파일을 찾을 수 없습니다. 학습을 먼저 진행해 주세요.")

    model.to(device)
    model.eval()

    # 전처리 (정규화 포함)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 파일 업로드
    file = st.file_uploader("테스트 이미지를 선택하세요", type=["jpg", "jpeg", "png"])

    if file is not None:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="업로드된 이미지", use_column_width=True)

        if st.button("분류하기"):
            with st.spinner("분석 중입니다..."):
                try:
                    # 텐서 변환 및 차원 추가
                    x = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(x)
                        probs = torch.softmax(outputs, dim=1)
                        top5_prob, top5_idx = torch.topk(probs, 5)

                    st.write("### Top-5 Predictions:")
                    for i in range(5):
                        idx = top5_idx[0][i].item()
                        cls_name = classes[idx]
                        prob = top5_prob[0][i].item()
                        
                        st.write(f"{i+1}. **{cls_name}** ({prob:.2%})")
                
                except Exception as e:
                    st.error("추론 중 오류가 발생했습니다. 이미지와 모델 파일을 다시 확인해 주세요.")

if __name__ == '__main__':
    main()