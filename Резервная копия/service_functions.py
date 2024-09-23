
############# Служебные функции для анализа keypoints_df #################

# Cтруктура данных Ключевые точки тела
import numpy as np
import pandas as pd


BODY = {
    "HEAD":{
    "NOSE": {'x': "NOSE_x", 'y': "NOSE_y"},
    "LEFT_EAR": {'x': "LEFT_EAR_x", 'y': "LEFT_EAR_y"},
    "RIGHT_EAR": {'x': "RIGHT_EAR_x", 'y': "RIGHT_EAR_y"}
    },
    "TORSO":{
    "LEFT_SHOULDER": {'x': "LEFT_SHOULDER_x", 'y': "LEFT_SHOULDER_y"},
    "RIGHT_SHOULDER": {'x': "RIGHT_SHOULDER_x", 'y': "RIGHT_SHOULDER_y"},
    "LEFT_HIP": {'x': "LEFT_HIP_x", 'y': "LEFT_HIP_y"},
    "RIGHT_HIP": {'x': "RIGHT_HIP_x", 'y': "RIGHT_HIP_y"}
    },
    "ARMS":{
    "LEFT_ELBOW": {'x': "LEFT_ELBOW_x", 'y': "LEFT_ELBOW_y"},
    "RIGHT_ELBOW": {'x': "RIGHT_ELBOW_x", 'y': "RIGHT_ELBOW_y"},
    "LEFT_WRIST": {'x': "LEFT_WRIST_x", 'y': "LEFT_WRIST_y"},
    "RIGHT_WRIST": {'x': "RIGHT_WRIST_x", 'y': "RIGHT_WRIST_y"}
    },
    "LEGS":{
    "LEFT_KNEE": {'x': "LEFT_KNEE_x", 'y': "LEFT_KNEE_y"},
    "RIGHT_KNEE": {'x': "RIGHT_KNEE_x", 'y': "RIGHT_KNEE_y"},
    "LEFT_ANKLE": {'x': "LEFT_ANKLE_x", 'y': "LEFT_ANKLE_y"},
    "RIGHT_ANKLE": {'x': "RIGHT_ANKLE_x", 'y': "RIGHT_ANKLE_y"},
    "LEFT_HEEL": {'x': "LEFT_HEEL_x", 'y': "LEFT_HEEL_y"},
    "RIGHT_HEEL": {'x': "RIGHT_HEEL_x", 'y': "RIGHT_HEEL_y"}
    }
}

KEY_POINTS = [coordinate for body_part in BODY.values() for body_part_point in body_part.values() for coordinate in body_part_point.values()]

def filter_rows_by_y_range(df, min_value=-0.2, max_value=1.2):
    """
    Фильтрует строки DataFrame, удаляя те, в которых хотя бы одно значение в столбцах,
    содержащих '_x', выходит за пределы заданного диапазона.

    Parameters:
    df : DataFrame
        Исходный DataFrame для фильтрации.
    min_value : float
        Минимальное допустимое значение.
    max_value : float
        Максимальное допустимое значение.

    Returns:
    DataFrame
        Отфильтрованный DataFrame.
    """
    # Выбираем только столбцы, которые содержат '_y'
    x_columns = [col for col in df.columns if '_y' in col]

    # Создаем маску для строк, где все значения в этих столбцах находятся в диапазоне
    mask = (df[x_columns] >= min_value).all(axis=1) & (df[x_columns] <= max_value).all(axis=1)

    # if has_internal_false(mask):
    #     return None

    # Возвращаем отфильтрованный df
    return mask

def get_coordinates(body_part, side="BOTH"):
    if side in ["LEFT", "RIGHT"]:
        filtered_body_part = {key: value for key, value in BODY[body_part].items() if side in key}
        return [coordinate for body_part_point in filtered_body_part.values() for coordinate in body_part_point.values()]
    else:
        return [coordinate for body_part_point in BODY[body_part].values() for coordinate in body_part_point.values()]

LEGS=get_coordinates("LEGS")
ARMS=get_coordinates("ARMS")
RIGHT_ARM = get_coordinates("ARMS","RIGHT")
HEAD = get_coordinates("HEAD")
TORSO = get_coordinates("TORSO")

# display(LEGS,'\n',ARMS,'\n',RIGHT_ARM,'\n',HEAD,'\n',TORSO)

# Найти в df num_col колонoк с максимальной амплитудой
def get_max_amplitude_col(df,num_col=1):

  num_columns=len(df.columns)
  # гарантируем, что не превысим индекс
  if num_col>num_columns: num_col=num_columns

  # Вычислить амплитуду (разницу между макс. и мин. значениями) для каждой колонки
  amplitudes = df.apply(lambda x: np.max(x) - np.min(x))

  # Отсортировать колонки по убыванию амплитуды и вернуть num_col названий столбцов с самыми блольшими амплитудами
  return amplitudes.sort_values(ascending=False).head(num_col).index

def get_max_energy_columns(df, columns, num_columns):
    # display('get_max_energy_columns: df',df)
    # print('get_max_energy_columns:columns, num_columns',columns, num_columns)

    """
    Возвращает имена столбцов из списка `columns` с максимальной энергией изменений.

    Энергия определяется как сумма квадратов отклонений от среднего значения в столбце.
    """
    energy = {}

    if columns==[]:
        columns=df.columns

    # Ограничиваем анализ только столбцами из списка `columns`
    for column in columns:
        if column in df.columns:
            # Вычитаем среднее значение из каждого элемента столбца
            deviations = df[column] - df[column].mean()

            # Вычисляем энергию изменений как сумму квадратов отклонений
            energy[column] = np.sum(deviations ** 2)

    # Сортируем столбцы по энергии и выбираем заданное количество
    sorted_columns = sorted(energy, key=energy.get, reverse=True)

    return sorted_columns[:num_columns]

# выбрать из списка колонок колонки по определенной оси (x,y,z)
def get_ax_columns(columns,ax):
  return [col for col in columns if '_'+ax in col]

# нормализовать данные в колонке df
def normalize_time_series(time_series):
    min = np.min(time_series)
    max = np.max(time_series)
    if max==min:
      return None
    return (time_series - min) / (max - min)

# Нормируем данные df_keypoints по глобальному мин/макс для унификации размеров тела
def normalize_keypoints(df_keypoints):
  # display(df_keypoints)

  #Удаляем тренды
  # df_keypoints=df_keypoints.apply(lambda x: x - np.polyval(np.polyfit(df_keypoints.index, x, 1), df_keypoints.index))

  # Находим минимальные и максимальные значения координат x и y
  x_columns = get_ax_columns(df_keypoints.columns,'x')
  y_columns = get_ax_columns(df_keypoints.columns,'y')
  # display(x_columns)
  # display(y_columns)

  min_x = df_keypoints[x_columns].min().min()
  max_x = df_keypoints[x_columns].max().max()
  min_y = df_keypoints[y_columns].min().min()
  max_y = df_keypoints[y_columns].max().max()

  # print('min_x, min_y, max_x, max_y', min_x, min_y, max_x, max_y)

  # Вычисляем ширину и высоту прямоугольника
  width = max_x - min_x
  height = max_y - min_y
  # print('width,height', width,height)

  # Нормируем координаты относительно прямоугольника

  df_normalized = pd.concat([
      df_keypoints[x_columns].apply(lambda x: (x - min_x) / width),
      df_keypoints[y_columns].apply(lambda y: (y - min_y) / height)
      ],axis=1)

  # display(df_normalized)

  return df_normalized

  ESSENTIAL_POINTS =['RIGHT_KNEE_y','LEFT_ELBOW_y', 'RIGHT_SHOULDER_y','LEFT_WRIST_y','NOSE_y','LEFT_WRIST_x','LEFT_ELBOW_x']
