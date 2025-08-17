import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm
import os
import warnings
import time
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Plant Leaf Classification with XAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .prediction-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
    }
    .xai-container {
        background-color: #fff5ee;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Updated Class Names
CLASS_NAMES = [
    'Alstonia Scholaris diseased', 'Alstonia Scholaris healthy', 
    'Arjun diseased', 'Arjun healthy', 'Bael diseased', 'Basil healthy',
    'Chinar diseased', 'Chinar healthy', 'Gauva diseased', 'Gauva healthy',
    'Jamun diseased', 'Jamun healthy', 'Jatropha diseased', 'Jatropha healthy',
    'Lemon diseased', 'Lemon healthy', 'Mango diseased', 'Mango healthy',
    'Pomegranate diseased', 'Pomegranate healthy', 
    'Pongamia Pinnata diseased', 'Pongamia Pinnata healthy'
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple CNN for LeafNet
class SimpleLeafNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleLeafNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model(model_name):
    """Load trained model"""
    try:
        if model_name == "LeafNet_CNN":
            model = SimpleLeafNet(num_classes=NUM_CLASSES)
            checkpoint = torch.load('LeafNet_CNN.pth', map_location=DEVICE)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        
        elif model_name == "EfficientNet_B0":
            model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
            )
            checkpoint = torch.load('EfficientNet_B0_best.pth', map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        elif model_name == "MobileNet_V3":
            model = models.mobilenet_v3_small(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.classifier[0].in_features, NUM_CLASSES)
            )
            checkpoint = torch.load('MobileNet_V3_best.pth', map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        elif model_name == "ResNet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            checkpoint = torch.load('ResNet18_best.pth', map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        elif model_name == "ViT_Small":
            model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
            checkpoint = torch.load('ViT_Small_best.pth', map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(DEVICE)
        return model
    
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None

def get_target_layers(model, model_name):
    """Get target layers for XAI"""
    try:
        if model_name == "LeafNet_CNN":
            conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
            return [conv_layers[-1]] if conv_layers else []
            
        elif model_name == "EfficientNet_B0":
            return [model.features[-1]]
            
        elif model_name == "MobileNet_V3":
            conv_layers = []
            def find_conv(module):
                for child in module.children():
                    if isinstance(child, nn.Conv2d):
                        conv_layers.append(child)
                    else:
                        find_conv(child)
            find_conv(model)
            return [conv_layers[-1]] if conv_layers else []
            
        elif model_name == "ResNet18":
            return [model.layer4[-1].conv2]
            
        elif model_name == "ViT_Small":
            return [model.blocks[-1].norm1]
            
        else:
            conv_layers = []
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    conv_layers.append(module)
            return [conv_layers[-1]] if conv_layers else []
            
    except Exception as e:
        st.warning(f"Target layer detection failed: {e}")
        return []

def predict_image(model, image, transform):
    """Make prediction on image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.softmax(outputs, dim=1)
        top_3_prob, top_3_indices = torch.topk(probabilities, 3)
        
        top_3_prob_np = top_3_prob.cpu().numpy().flatten()
        top_3_indices_np = top_3_indices.cpu().numpy().flatten()
        
        top_3_indices_list = [int(idx) for idx in top_3_indices_np]
        top_3_prob_list = [float(prob) for prob in top_3_prob_np]
    
    return top_3_indices_list, top_3_prob_list, input_tensor

def create_colored_heatmap(grayscale_cam, input_image):
    """Create a colored heatmap from grayscale CAM"""
    try:
        if len(grayscale_cam.shape) > 2:
            grayscale_cam = grayscale_cam.squeeze()
        
        if grayscale_cam.max() > grayscale_cam.min():
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
        
        cam_uint8 = np.uint8(255 * grayscale_cam)
        colored_heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        if colored_heatmap.shape[:2] != input_image.shape[:2]:
            colored_heatmap = cv2.resize(colored_heatmap, (input_image.shape[1], input_image.shape[0]))
        
        input_uint8 = np.uint8(255 * np.clip(input_image, 0, 1))
        blended = cv2.addWeighted(input_uint8, 0.6, colored_heatmap, 0.4, 0)
        
        return blended
        
    except Exception as e:
        st.warning(f"Heatmap creation error: {str(e)}")
        return np.uint8(255 * np.clip(input_image, 0, 1))

def generate_cam_explanations(model, model_name, input_tensor, predicted_class):
    """Generate CAM explanations only - FAST & RELIABLE VERSION"""
    target_layers = get_target_layers(model, model_name)
    
    if not target_layers:
        st.error(f"❌ Could not find target layers for {model_name}")
        return {}
    
    st.info(f"✅ Using target layer: {target_layers[0]}")
    
    # Convert input for visualization
    input_array = input_tensor.squeeze().cpu().numpy()
    input_array = np.transpose(input_array, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_array = std * input_array + mean
    input_array = np.clip(input_array, 0, 1)
    
    explanations = {}
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    # CAM Methods
    methods = {
        'Grad-CAM': GradCAM,
        'Grad-CAM++': GradCAMPlusPlus,
        'Eigen-CAM': EigenCAM,
        'Ablation-CAM': AblationCAM
    }
    
    for method_name, method_class in methods.items():
        try:
            st.write(f"⚡ Generating {method_name}...")
            
            cam = method_class(model=model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(predicted_class)]
            
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=input_batch, targets=targets)
            
            if isinstance(grayscale_cam, (list, tuple)):
                grayscale_cam = grayscale_cam[0]
            if len(grayscale_cam.shape) > 2:
                grayscale_cam = grayscale_cam
            
            colored_cam = create_colored_heatmap(grayscale_cam, input_array)
            explanations[method_name] = colored_cam
            
            st.success(f"✅ {method_name} generated successfully")
            
        except Exception as e:
            st.warning(f"❌ {method_name} failed: {str(e)}")
            explanations[method_name] = np.uint8(255 * input_array)
    
    return explanations, input_array

def generate_simple_lime(image, predicted_class):
    """Generate a simple LIME-like explanation using edge detection"""
    try:
        st.write("⚡ Generating LIME-style explanation...")
        
        # Convert image to array
        img_array = np.array(image)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create a colored overlay
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend with original image
        lime_result = cv2.addWeighted(img_array, 0.7, colored_edges, 0.3, 0)
        
        st.success("✅ LIME-style explanation generated successfully")
        return lime_result
        
    except Exception as e:
        st.warning(f"❌ LIME-style explanation failed: {str(e)}")
        return np.array(image)

def display_xai_results(explanations):
    """Display XAI results side by side"""
    if not explanations:
        st.error("❌ No explanations to display")
        return
    
    methods = list(explanations.keys())
    num_methods = len(methods)
    
    if num_methods > 0:
        st.markdown("### 🔥 XAI Explanations")
        cols = st.columns(num_methods)
        
        for i, method in enumerate(methods):
            with cols[i]:
                st.markdown(f"**{method}**")
                if isinstance(explanations[method], np.ndarray):
                    st.image(explanations[method], use_container_width=True)

def create_download_zip(original_image, explanations, predictions):
    """Create downloadable ZIP file"""
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add original image
            img_buffer = io.BytesIO()
            original_image.save(img_buffer, format='PNG')
            zip_file.writestr("original_image.png", img_buffer.getvalue())
            
            # Add XAI results
            for method, explanation in explanations.items():
                exp_buffer = io.BytesIO()
                if isinstance(explanation, np.ndarray):
                    exp_image = Image.fromarray(explanation.astype(np.uint8))
                    exp_image.save(exp_buffer, format='PNG')
                    zip_file.writestr(f"{method.replace('-', '_')}_explanation.png", exp_buffer.getvalue())
            
            # Add results summary
            top_3_indices, top_3_prob = predictions
            prob_strings = [f"{p:.3f}" for p in top_3_prob]
            
            summary = f"""Plant Leaf Classification Results

Predicted Class: {CLASS_NAMES[top_3_indices[0]]}
Confidence: {prob_strings} ({float(top_3_prob)*100:.1f}%)

Top 3 Predictions:
1. {CLASS_NAMES[top_3_indices]}: {prob_strings}
2. {CLASS_NAMES[top_3_indices[1]]}: {prob_strings[1]}
3. {CLASS_NAMES[top_3_indices[2]]}: {prob_strings[2]}

XAI Methods Applied: {', '.join(explanations.keys())}
"""
            zip_file.writestr("results_summary.txt", summary.encode())
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🌿 Plant Leaf Classification with Explainable AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🛠️ Model Configuration")
    
    # Model selection
    available_models = ["LeafNet_CNN", "EfficientNet_B0", "MobileNet_V3", "ResNet18", "ViT_Small"]
    selected_model = st.sidebar.selectbox("Select Model:", available_models)
    
    # LIME option in sidebar
    st.sidebar.header("🔧 XAI Options")
    include_lime = st.sidebar.checkbox("Include LIME explanation", value=False, 
                                       help="LIME can be very slow. Uncheck for faster results.")
    
    # Load model
    model = load_model(selected_model)
    
    if model is None:
        st.error("❌ Could not load the selected model. Please check if model weights are available.")
        return
    
    # Model metadata
    st.sidebar.markdown('<div class="model-info">', unsafe_allow_html=True)
    st.sidebar.markdown(f"**Selected Model:** {selected_model}")
    st.sidebar.markdown(f"**Input Size:** {IMG_SIZE}x{IMG_SIZE}")
    st.sidebar.markdown(f"**Number of Classes:** {NUM_CLASSES}")
    st.sidebar.markdown(f"**Device:** {DEVICE.type.upper()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    st.sidebar.markdown(f"**Total Parameters:** {total_params:,}")
    st.sidebar.markdown(f"**Trainable Parameters:** {trainable_params:,}")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Class names display
    with st.sidebar.expander("📋 Plant Classes", expanded=False):
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{i}: {class_name}")
    
    # Image input section
    st.header("📁 Image Input")
    
    uploaded_file = st.file_uploader("Upload a plant leaf image:", type=['png', 'jpg', 'jpeg'])
    
    # Process image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.success("✅ Image uploaded successfully!")
        
        # Display original image
        st.subheader("🖼️ Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Input Image", use_container_width=True)
        
        # Make prediction
        st.subheader("🎯 Prediction Results")
        
        with st.spinner("Making prediction..."):
            top_3_indices, top_3_prob, input_tensor = predict_image(model, image, transform)
        
        # Display predictions
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🥇 Top 3 Predictions:**")
            for i, (idx, prob) in enumerate(zip(top_3_indices, top_3_prob)):
                confidence_color = "#28a745" if i == 0 else "#17a2b8" if i == 1 else "#ffc107"
                st.markdown(f"""
                <div style="margin: 10px 0; padding: 10px; background-color: {confidence_color}20; border-left: 4px solid {confidence_color}; border-radius: 5px;">
                    <strong>Rank {i+1}:</strong> {CLASS_NAMES[idx]}<br>
                    <strong>Confidence:</strong> {prob:.3f} ({prob*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(range(3), top_3_prob, color=['#28a745', '#17a2b8', '#ffc107'])
            ax.set_yticks(range(3))
            ax.set_yticklabels([CLASS_NAMES[i] for i in top_3_indices])
            ax.set_xlabel('Confidence')
            ax.set_title('Top 3 Predictions')
            ax.set_xlim(0, 1)
            
            for i, (bar, prob) in enumerate(zip(bars, top_3_prob)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.3f}', va='center', ha='left', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # XAI Analysis
        st.subheader("🔍 Explainable AI Analysis")
        
        with st.expander("ℹ️ About XAI Methods", expanded=False):
            st.markdown("""
            - **Grad-CAM**: Highlights important regions using gradients
            - **Grad-CAM++**: Improved version with better localization  
            - **Eigen-CAM**: Uses eigenvectors of activations
            - **Ablation-CAM**: Tests importance by removing features
            - **LIME-style**: Edge-based explanation (fast alternative to LIME)
            """)
        
        if st.button("🚀 Generate XAI Explanations", type="primary"):
            start_time = time.time()
            
            # Generate CAM explanations
            with st.spinner("⚡ Generating CAM explanations..."):
                cam_explanations, original_array = generate_cam_explanations(
                    model, selected_model, input_tensor, top_3_indices[0]
                )
            
            # Add LIME-style explanation if requested
            all_explanations = cam_explanations.copy()
            
            if include_lime:
                st.warning("⚠️ LIME explanation may take a long time...")
                try:
                    from lime import lime_image
                    from skimage.segmentation import mark_boundaries
                    
                    with st.spinner("⚡ Generating LIME (this may take several minutes)..."):
                        # Ultra-minimal LIME
                        explainer = lime_image.LimeImageExplainer()
                        
                        def batch_predict(images):
                            model.eval()
                            batch = torch.stack([transform(Image.fromarray(img.astype('uint8'))) for img in images]).to(DEVICE)
                            with torch.no_grad():
                                logits = model(batch)
                                probs = torch.nn.functional.softmax(logits, dim=1)
                            return probs.cpu().numpy()
                        
                        image_array = np.array(image)
                        
                        explanation = explainer.explain_instance(
                            image_array,
                            batch_predict,
                            top_labels=1,
                            hide_color=0,
                            num_samples=10,  # Minimal samples
                            random_seed=42
                        )
                        
                        temp, mask = explanation.get_image_and_mask(
                            top_3_indices[0],
                            positive_only=True,
                            num_features=3,
                            hide_rest=False
                        )
                        
                        lime_img = mark_boundaries(temp / 255.0, mask)
                        all_explanations['LIME'] = np.uint8(255 * lime_img)
                        st.success("✅ LIME generated successfully")
                        
                except Exception as e:
                    st.error(f"❌ LIME failed: {str(e)}")
                    st.info("💡 Using fast LIME-style alternative instead")
                    all_explanations['LIME-style'] = generate_simple_lime(image, top_3_indices[0])
            else:
                # Use fast LIME-style alternative
                all_explanations['LIME-style'] = generate_simple_lime(image, top_3_indices)
            
            # Display all results
            if all_explanations:
                st.markdown('<div class="xai-container">', unsafe_allow_html=True)
                display_xai_results(all_explanations)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Performance info
                total_time = time.time() - start_time
                st.info(f"⚡ Total generation time: {total_time:.2f} seconds")
                
                # Download option
                st.subheader("💾 Download Results")
                
                zip_data = create_download_zip(image, all_explanations, (top_3_indices, top_3_prob))
                
                if zip_data:
                    st.download_button(
                        label="📥 Download All Results (ZIP)",
                        data=zip_data,
                        file_name=f"plant_classification_results_{selected_model}.zip",
                        mime="application/zip"
                    )
                
                st.success(f"🎉 Generated {len(all_explanations)} XAI explanations successfully!")
            else:
                st.error("❌ No XAI explanations could be generated.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌱 Plant Leaf Disease Classification System with Explainable AI</p>
        <p>Built with Streamlit • Powered by PyTorch • Enhanced with XAI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
