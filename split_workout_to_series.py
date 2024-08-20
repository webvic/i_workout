
# Ищем точки разбиения видео зарядки на однородные серии упражнений
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from service_functions import get_max_energy_columns
import ruptures as rpt
from scipy.signal import find_peaks

def get_change_points(df_keypoints):
  num_max_energy_cols=5
  # отбираем ряды с макс удельной мощностью
  max_energy_cols=get_max_energy_columns(df_keypoints,[],num_max_energy_cols)

  # Инициализация алгоритма обнаружения изменений
  algo = rpt.Window(width=150, model="rbf").fit(df_keypoints[max_energy_cols].values) # Оценка 4
  # algo = rpt.Binseg(model="l2").fit(df_keypoints.values) # Оценка 4-
  # algo = rpt.Pelt(model="l2").fit(df_keypoints.values) # Оценка 3-
  # algo = rpt.BottomUp(model="l2").fit(df_keypoints.values) # Оценка 3-
  # algo = rpt.Dynp(model="l2", min_size=30, jump=5).fit(df_keypoints.values) #Требует к-ва брейкпойнтов
  # algo = rpt.KernelCPD(kernel="linear").fit(df_keypoints.values) # Оценка 3-

  # Detect change points
  change_points = algo.predict(pen=0)  # Penalty can be adjusted

  # Display the results
  # rpt.display(df_keypoints.values, change_points)
  # plt.show()

  # change_points = algo.predict()
  # print('len(change_points)',len(change_points), change_points)

  if change_points[0] != 0:
      change_points.insert(0, 0)

  if change_points[-1] == len(df_keypoints):
      change_points[-1] -= 1
  else:
      change_points.append(len(df_keypoints)-1)

  # print('len(change_points)',len(change_points), change_points)

  return change_points, max_energy_cols

from sklearn.cluster import KMeans

def zero_crossing_rate(signal):
    # Subtract mean to center the signal
    centered_signal = signal - np.mean(signal)
    # Calculate the number of zero crossings
    zero_crossings = np.sum(np.diff(np.sign(centered_signal)) != 0)
    # Return the rate of zero crossings per unit length
    return zero_crossings / len(signal)

def calculate_segment_metrics(segment):
    length = len(segment)
    # print('calculate_segment_metrics: length, segment', length, segment)

    # Calculate linear trends and detrend the signal
    coeffs = [np.polyfit(range(length), seg, 1) for seg in segment.T]
    detrended_segment = segment - np.outer(np.arange(length), [coeff[0] for coeff in coeffs])
    detrended_segment -= np.array([coeff[1] for coeff in coeffs]).reshape(1, -1)

    # Total trend
    total_trend = np.sum(np.abs([coeff[0] for coeff in coeffs]))

    # Cyclicity: Count peaks above half of the maximum amplitude
    cyclicity = 0
    for seg in detrended_segment.T:
        max_amp = np.max(seg)
        peaks, _ = find_peaks(seg, height=max_amp / 2)
        cyclicity += len(peaks)

    # Zero Crossing Rate for each signal (normalized by segment length)
    zero_crossings = [zero_crossing_rate(seg) for seg in detrended_segment.T]
    average_zero_crossing_rate = np.mean(zero_crossings)

    # Return metrics
    return {
        'length': length,
        'total_trend': total_trend,
        'cyclicity': cyclicity,
        'zero_crossing_rate': average_zero_crossing_rate
    }

def classify_segments(df, segment_indices, expert_labels=None):
    metrics = []
    for i in range(len(segment_indices) - 1):
        start, end = segment_indices[i], segment_indices[i+1]
        segment_df = df.iloc[start:end].to_numpy()

        # Calculate segment metrics
        metrics_data = calculate_segment_metrics(segment_df)
        metrics_data['indices'] = f"{start}:{end}"

        metrics.append(metrics_data)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.set_index(metrics_df.pop('indices'))

    # Display raw metrics
    # print('Raw Metrics:')
    # display(metrics_df)

    # Normalize the metrics
    scaler = MinMaxScaler()
    metrics_df[['total_trend', 'cyclicity']] = scaler.fit_transform(metrics_df[['total_trend', 'cyclicity']])

    # Optimization function: we want high cyclicity and low total trend
    # Adjust weights as needed for your specific dataset
    metrics_df['score'] = metrics_df['cyclicity'] - metrics_df['total_trend']

    # Threshold for classification
    threshold = 0 # You can adjust this based on analysis
    metrics_df['classification'] = metrics_df['score'].apply(lambda x: 'Good' if x > threshold else 'Bad')

    is_strong_segment = metrics_df['classification'] == 'Good'

    return is_strong_segment, metrics_df

# Тестовый блок
# POINTS=ESSENTIAL_POINTS
# POINTS=KEY_POINTS
POINTS=['NOSE_y','RIGHT_WRIST_y','RIGHT_ELBOW_y','RIGHT_WRIST_x']
