
############### Разбиение серии упражнений на сэмплы ################

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch, find_peaks
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, hilbert

from service_functions import get_max_energy_columns

def get_serie_breaking_points(segment_df, fs=1.0, prominence=0.3, height_factor=0.7, smoothing_window=5, poly_order=2):
    """
    Анализирует временные ряды, усредняет их, находит гармонику с максимальной мощностью, определяет пики.
    """
    # Суживаем анализ точек разрыва сэмплов до N наиболее энергичных рядов
    segment_df = segment_df[get_max_energy_columns(segment_df,[],1)]

    # Сглаживаем и усредняем временные ряды
    smoothed_df = segment_df.apply(lambda x: savgol_filter(x, window_length=smoothing_window, polyorder=poly_order), axis=0)
    average_signal = smoothed_df.mean(axis=1)

    # Вычисляем спектральную плотность мощности (PSD)
    freqs, psd = welch(average_signal, fs=fs, nperseg=min(256, len(average_signal)))

    # Находим частоту с максимальной мощностью
    max_freq_index = np.argmax(psd)
    max_freq = freqs[max_freq_index]
    period = int(fs / max_freq) if max_freq > 0 else len(average_signal)  # Период гармоники

    # Генерируем гармоническую волну с максимальной мощностью
    harmonic_wave = np.cos(2 * np.pi * max_freq * np.arange(len(average_signal)) / fs)

    # Поиск позитивных пиков в гармонической волне
    harmonic_peaks, _ = find_peaks(harmonic_wave)

    # Добавляем 0 в список индексов, если он не был добавлен
    peak_indices = list(average_signal.index[harmonic_peaks])
    if 0 not in peak_indices:
        peak_indices.insert(0, average_signal.index[0])

    # Add the last index if the last segment is longer than 2/3 of the period
    if len(harmonic_peaks) == 0:
        peak_indices = [average_signal.index[0], average_signal.index[-1]]
    else:
        if len(average_signal) - harmonic_peaks[-1] >= 2/3 * period:
            peak_indices.append(average_signal.index[-1])

    return peak_indices
