import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage import measure, morphology
import matplotlib.pyplot as plt

class AdaptiveMicroplasticDetector:
    def __init__(self):
        self.pixel_size = 5.35
        
    def preprocess_image(self, image_array):
        """Process numpy array instead of file path"""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        return gray, image_array
    
    def detect_particles_adaptive(self, gray_image):
        results = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Strategy 1: Otsu
        blurred1 = cv2.GaussianBlur(gray_image, (3, 3), 0)
        _, binary1 = cv2.threshold(blurred1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary1 = cv2.morphologyEx(binary1, cv2.MORPH_OPEN, kernel, iterations=1)
        contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid1 = [c for c in contours1 if 3 < cv2.contourArea(c) < 800]
        results.append(('Otsu', valid1, binary1))
        
        # Strategy 2: Threshold 50
        blurred2 = cv2.GaussianBlur(gray_image, (3, 3), 0)
        _, binary2 = cv2.threshold(blurred2, 50, 255, cv2.THRESH_BINARY)
        binary2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, kernel, iterations=1)
        contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid2 = [c for c in contours2 if 2 < cv2.contourArea(c) < 600]
        results.append(('Threshold50', valid2, binary2))
        
        # Strategy 3: Threshold 35
        blurred3 = cv2.GaussianBlur(gray_image, (3, 3), 0)
        _, binary3 = cv2.threshold(blurred3, 35, 255, cv2.THRESH_BINARY)
        binary3 = cv2.morphologyEx(binary3, cv2.MORPH_OPEN, kernel, iterations=1)
        contours3, _ = cv2.findContours(binary3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid3 = [c for c in contours3 if 2 < cv2.contourArea(c) < 500]
        results.append(('Threshold35', valid3, binary3))
        
        # Strategy 4: Adaptive
        blurred4 = cv2.GaussianBlur(gray_image, (5, 5), 0)
        binary4 = cv2.adaptiveThreshold(blurred4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, -3)
        binary4 = cv2.morphologyEx(binary4, cv2.MORPH_OPEN, kernel, iterations=1)
        contours4, _ = cv2.findContours(binary4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid4 = [c for c in contours4 if 3 < cv2.contourArea(c) < 600]
        results.append(('Adaptive', valid4, binary4))
        
        best_method, best_contours, best_binary = self.select_best_method(results)
        return best_contours, best_binary, best_method
    
    def select_best_method(self, results):
        scores = []
        for method_name, contours, binary in results:
            if len(contours) == 0:
                scores.append((0, method_name, contours, binary))
                continue
            
            areas = [cv2.contourArea(c) for c in contours]
            avg_area = np.mean(areas)
            std_area = np.std(areas)
            count = len(contours)
            
            count_score = 1.0
            if count < 5:
                count_score = 0.3
            elif count > 500:
                count_score = 0.5
            
            size_consistency = 1.0 / (1.0 + std_area / max(avg_area, 1))
            size_score = 1.0 if 5 < avg_area < 200 else 0.5
            total_score = count_score * size_consistency * size_score
            scores.append((total_score, method_name, contours, binary))
        
        scores.sort(reverse=True)
        _, best_method, best_contours, best_binary = scores[0]
        return best_method, best_contours, best_binary
    
    def analyze_array(self, image_array):
        """Analyze numpy array for Streamlit"""
        gray, original = self.preprocess_image(image_array)
        contours, binary, method = self.detect_particles_adaptive(gray)
        
        particle_count = len(contours)
        areas = [cv2.contourArea(c) for c in contours]
        
        result = original.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        if particle_count <= 50:
            for i, contour in enumerate(contours, 1):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result, str(i), (cx-5, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return {
            'count': particle_count,
            'method': method,
            'binary': binary,
            'result': result,
            'areas': areas
        }

# STREAMLIT APP
st.set_page_config(page_title="Microplastic Detector", page_icon="🔬", layout="wide")

st.title("🔬 Microplastic Detection App")
st.write("Upload fluorescence microscopy images to detect microplastic particles")

detector = AdaptiveMicroplasticDetector()

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Analyzing..."):
            results = detector.analyze_array(img_array)
            
            with col2:
                st.subheader("Detection Results")
                st.image(results['result'], use_column_width=True)
            
            st.success("Analysis Complete!")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Particle Count", results['count'])
            with m2:
                st.metric("Method", results['method'])
            with m3:
                if results['areas']:
                    st.metric("Avg Size", f"{np.mean(results['areas']):.1f} px")
            
            st.subheader("Detection Mask")
            st.image(results['binary'], use_column_width=True)

st.sidebar.header("About")
st.sidebar.info("Detects microplastic particles in fluorescence microscopy images")