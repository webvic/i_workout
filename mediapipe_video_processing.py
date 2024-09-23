# Инициализируем структуру ключевых точек тела
import cv2
import pandas as pd
import mediapipe as mp



def initialize_key_points():
  keypoints_data = {

      "NOSE_x": [], "NOSE_y": [],
      "LEFT_EYE_INNER_x": [], "LEFT_EYE_INNER_y": [],
      "LEFT_EYE_x": [], "LEFT_EYE_y": [],
      "LEFT_EYE_OUTER_x": [], "LEFT_EYE_OUTER_y": [],
      "RIGHT_EYE_INNER_x": [], "RIGHT_EYE_INNER_y": [],
      "RIGHT_EYE_x": [], "RIGHT_EYE_y": [],
      "RIGHT_EYE_OUTER_x": [], "RIGHT_EYE_OUTER_y": [],
      "LEFT_EAR_x": [], "LEFT_EAR_y": [],
      "RIGHT_EAR_x": [], "RIGHT_EAR_y": [],
      "MOUTH_LEFT_x": [], "MOUTH_LEFT_y": [],
      "MOUTH_RIGHT_x": [], "MOUTH_RIGHT_y": [],
      "LEFT_SHOULDER_x": [], "LEFT_SHOULDER_y": [],
      "RIGHT_SHOULDER_x": [], "RIGHT_SHOULDER_y": [],
      "LEFT_ELBOW_x": [], "LEFT_ELBOW_y": [],
      "RIGHT_ELBOW_x": [], "RIGHT_ELBOW_y": [],
      "LEFT_WRIST_x": [], "LEFT_WRIST_y": [],
      "RIGHT_WRIST_x": [], "RIGHT_WRIST_y": [],
      "LEFT_PINKY_x": [], "LEFT_PINKY_y": [],
      "RIGHT_PINKY_x": [], "RIGHT_PINKY_y": [],
      "LEFT_INDEX_x": [], "LEFT_INDEX_y": [],
      "RIGHT_INDEX_x": [], "RIGHT_INDEX_y": [],
      "LEFT_THUMB_x": [], "LEFT_THUMB_y": [],
      "RIGHT_THUMB_x": [], "RIGHT_THUMB_y": [],
      "LEFT_HIP_x": [], "LEFT_HIP_y": [],
      "RIGHT_HIP_x": [], "RIGHT_HIP_y": [],
      "LEFT_KNEE_x": [], "LEFT_KNEE_y": [],
      "RIGHT_KNEE_x": [], "RIGHT_KNEE_y": [],
      "LEFT_ANKLE_x": [], "LEFT_ANKLE_y": [],
      "RIGHT_ANKLE_x": [], "RIGHT_ANKLE_y": [],
      "LEFT_HEEL_x": [], "LEFT_HEEL_y": [],
      "RIGHT_HEEL_x": [], "RIGHT_HEEL_y": [],
      "LEFT_FOOT_INDEX_x": [], "LEFT_FOOT_INDEX_y": [],
      "RIGHT_FOOT_INDEX_x": [], "RIGHT_FOOT_INDEX_y": []

  }
  return keypoints_data

# оцифровываем видео - получаем ключевые точки
def get_key_points_from_video(path_to_video):

  # Создание объекта для захвата видео
  cap = cv2.VideoCapture(path_to_video)
  # Проверка успешности операции
  if not cap.isOpened():
      print("Ошибка: Невозможно открыть видеофайл.")
  else:
      print(f"Видеофайл {path_to_video} успешно открыт.")

  #Создаем экземпляр модели BlazePose исходно настроенной на сложный поиск 33 точек одного человека
  # mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
  # Инициализируем таблицу временных рядов по ключевым точкам
  keypoints_data=initialize_key_points()

  # Получаем тип кодека и расширение
  # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
  # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)

  # print(f"Ширина: {width}, Высота: {height}, FPS: {fps}, Ext: {fourcc}")

  if not cap.isOpened():
      print("Ошибка при открытии видео файла.")
      cap.release()
      pose.close()
      exit()

  # Считываем видео и извлекаем позы

  ret, frame = cap.read()
  time=0
  frames_count=0
  resized_width=200

  while ret:

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:

        # Сбор данных ключевых точек
        for landmark in mp_pose.PoseLandmark:
            keypoints_data[f"{landmark.name}_x"].append(results.pose_landmarks.landmark[landmark].x)
            keypoints_data[f"{landmark.name}_y"].append(results.pose_landmarks.landmark[landmark].y)

        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # # масштабируем кадры до 200 точек ширины для экономии памяти
        # keypoints_data["Frames"].append(resize_frame_proportionally(frame, resized_width))
        # keypoints_data["Time"].append(time)

    time+=1/fps
    frames_count+=1

    ret, frame = cap.read()

  len_video=frames_count
  # print('len_video', len_video)

  pose.close()
  cap.release()

  # display('get_key_points_from_video:keypoints_data',keypoints_data)

  # Преобразуем данные в df
  df_keypoints=pd.DataFrame(keypoints_data)
  # display('get_key_points_from_video: df_keypoints', df_keypoints)

  return df_keypoints, time

# # Тестовый блок
# text_path='''
# /content/drive/MyDrive/Видео движений #2/video_2024-06-27_12-48-46.mp4
# '''

# paths_to_video = text_path.strip().splitlines()

# for path_to_video in paths_to_video:

#   # анализируем видео
#   df_keypoints = get_key_points_from_video(path_to_video)

#   # выделяем видео данные в отдельный df
#   df_video=df_keypoints[['Frames','Time']]

#   # отбираем только важные 17 точек из исходных 33
#   df_keypoints=df_keypoints[KEY_POINTS]

#   # display('df_keypoints',df_keypoints)

#   # display('df_keypoints, df_video', df_keypoints, df_video)

