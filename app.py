
################## Модули, Библиотеки и стримлит #####################

# !pip install streamlit

# Модули
!pip install imageio[ffmpeg]
!pip install opencv-python-headless --upgrade
!pip install ffmpeg-python

# Mediapipe
!pip install mediapipe
# !pip install protobuf==3.20.3

# анализ точек разрыва и квалификация серий
!pip install scikit-learn-extra
!pip install ruptures

# функция сравнения временных рядов DTW
!pip install tslearn

# Библиотеки
# import streamlit as st
from io import BytesIO
import cv2
import mediapipe as mp
from google.colab.patches import cv2_imshow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.signal as signal
from scipy.signal import medfilt
from scipy import fftpack
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import AgglomerativeClustering
import ruptures as rpt
from scipy.signal import welch, find_peaks
from sklearn_extra.cluster import KMedoids
from scipy.stats import entropy
from tslearn.metrics import dtw
import imageio
import io
from IPython.display import HTML, display
from IPython.display import Image as IPImage
import base64
from base64 import b64encode
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from PIL import Image
import uuid

warnings.filterwarnings("ignore")

# def main():
#     st.title("Video Analytics Application")

#     uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

#     if uploaded_file is not None:
#         video_bytes = uploaded_file.read()
#         process_video(BytesIO(video_bytes))

# if __name__ == '__main__':
#     main()
