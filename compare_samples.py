
################# Модуль вычисления близости между рядами #####################

import os
from cv2 import distanceTransformWithLabels
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw

from service_functions import BODY, KEY_POINTS, get_coordinates, get_max_energy_columns, normalize_keypoints


def cyclic_shift(df, shift):
    return df.apply(lambda x: np.roll(x, shift))

def shift_sample(df1, df2, columns):
    # Приводим DataFrame к одинаковой длине
    min_length = min(len(df1), len(df2))
    df1 = df1.iloc[:min_length]
    df2 = df2.iloc[:min_length]

    max_corr = -np.inf  # Начальное значение для максимальной корреляции
    best_shift = 0      # Начальное значение для лучшего сдвига

    # Циклический сдвиг для всей группы столбцов
    for shift in range(len(df2)):
        # Сдвигаем df2 с использованием функции cyclic_shift
        shifted_df2 = cyclic_shift(df2[columns], shift)

        # Вычисляем корреляцию для группы столбцов
        combined_correlation = 0
        for column in columns:
            if column in df1.columns and column in df2.columns:
                correlation = df1[column].corr(shifted_df2[column])
                if not np.isnan(correlation):
                    combined_correlation += correlation

        # Проверяем, является ли текущая корреляция максимальной
        if combined_correlation > max_corr:
            max_corr = combined_correlation
            best_shift = shift

    # Сдвигаем df2 на оптимальное количество шагов
    shifted_df2 = cyclic_shift(df2, best_shift)

    return shifted_df2

# Функция для подсчета расстояний между одноименными столбцами в однородных df
def DTW_compare(df1, df2, columns=[],num_compare_cols=4):
  # print('DTW_compare: columns',columns)

  # Используем по 4 точки для анализа каждой части тела
  num_compare_cols=4

  # print('def DTW_compare(df1, df2, columns=[]):',columns)
  # display('df1.shape, df2.shape', df1.shape, df2.shape)

  if len(columns)==0: columns=df1.columns.to_list()

  num_plots=num_compare_cols
  # fig, axes = plt.subplots(2, num_plots, figsize=(3*num_plots, 3))

  distances =[]

  max_energy_cols=get_max_energy_columns(df1,columns, num_compare_cols)

  # print('DTW_compare: max_energy_cols',max_energy_cols)

  for i, col in enumerate(max_energy_cols):

      # Рассчитываем расстояние методом DTW
      distances.append(dtw(df1[col], df2[col]))

  return np.array(distances).mean()

def compare_samples(df1, df2):
  # Количество столбцов для оптимизации методом регрессии
  num_reg_optim_cols = 5
  # print('def compare_samples(df1, df2):')

  # Удалим ненужные столбцы
  df1, df2 = df1[KEY_POINTS],df2[KEY_POINTS ]

  # Нормализуем
  df1, df2 = normalize_keypoints(df1),normalize_keypoints(df2)

  max_energy_cols=get_max_energy_columns(df1,[],num_reg_optim_cols)
  # print('compare_samples: max_energy_cols',max_energy_cols)

  # visualize_column_pairs(df1, df2, max_energy_cols)
  df2 = shift_sample (df1,df2,max_energy_cols)
  # print('compare_samples: После сдвига')
  # visualize_column_pairs(df1, df2, max_energy_cols)

  # Посчитаем близость сэмплов  с учетом весов рядов для каждой части тела
  eval_poze={}
  for body_part in BODY:
    # print('eval_poze[body_part]=DTW_compare(df1,df2,get_coordinates(body_part)):', body_part)
    eval_poze[body_part]=DTW_compare(df1,df2,get_coordinates(body_part))

  eval_poze['GENERAL'] = sum(eval_poze.values()) / len(eval_poze)
  # print('compare_samples: eval_poze',eval_poze)

  return eval_poze

def sample_classification(df_instructor,df_pupil,path_to_instructor_dir):
  # print('def sample_classification(df_pupil)')

  indicator=[]
  INDICES=['GENERAL']+list(BODY.keys())
  # print(INDICES)

  for index, row in df_instructor.iterrows():
    # считываем файл со схемой движения инструктора
    instructor_sample_df = pd.read_csv(os.path.join(path_to_instructor_dir, row['Ссылка на схему']))

    # display('sample_classification: instructor_sample_df, df_pupil',instructor_sample_df, df_pupil)
    # display(compare_samples(instructor_sample_df, df_pupil))
    # print('sample_classification: Сравниваем сэмпл с движением инструктора',row['Упражнение'],row['Вид'])
    indicator.append(compare_samples(instructor_sample_df, df_pupil))
    indicator[-1]['Упражнение']=row['Упражнение']
    indicator[-1]['Вид']=row['Вид']

  # print('sample_classification: Цикл пройден')

  df_indicator=pd.DataFrame(indicator)

  # display('sample_classification: df_indicator', df_indicator)

  scaler = MinMaxScaler()

  df_scaled=pd.DataFrame(scaler.fit_transform(df_indicator[INDICES]),columns=INDICES)

  # display('sample_classification: Нормированный', df_scaled)

  # Находим в нормированной таблице индекс мин показателя general и возвращаем строку индикатора с измеренными параметрами]
  min_general_row = df_indicator.loc[df_scaled['GENERAL'].idxmin()]
  # print('sample_classification ----- Конец работы -------- : min_general_row', min_general_row)

  # print('sample_classification: min_general_row', min_general_row)

  # возвращаем строку df с результатами данного сэмпла
  return min_general_row

# Преобразование оценки 0-1 -> 5-1
def convert_score(score):
    if score < 0.45:
      return 5
    elif score < 0.8:
      return 4
    elif score < 0.95:
      return 3
    return 2

# Сравнение всех сэмплов с образцами инструктора
# Разрезаем ряды ученика на сэмплы и квалифицируем каждый сэмпл и вычисляем среднюю оценку
def eval_pupi_df(df_segment, df_instructor, break_points,path_to_instructor_dir):
    # display('len(df_segment),df_segment, df_instructor, breaking_points', len(df_segment),df_segment, df_instructor, break_points)
    # png_paths = []
    # scores = []
    samples_num=len(break_points)-1
    result_clasification=[]
    # Цикл по семплам серии
    for i in range(samples_num):

        # Вырезаем семплы из df ученика
        start_point=break_points[i]
        end_point=break_points[i+1]-1
        # print('eval_pupil: start_point:end_point', start_point, end_point)

        df_pupil = df_segment.loc[start_point:end_point].reset_index(drop=True)

        # Получаем классификацию для семпла ученика
        result_clasification.append(sample_classification(df_instructor, df_pupil,path_to_instructor_dir).to_frame().T)
    
    serie_result_df = pd.concat(result_clasification)
    # display('eval_pupi_df: serie_result_df',serie_result_df)

    # # combined_path = combine_png_files(png_paths)
    # all_score=convert_score(np.array(scores).mean())

    return serie_result_df  
