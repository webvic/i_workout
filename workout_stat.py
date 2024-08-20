
import pandas as pd
import numpy as np
from compare_samples import convert_score


# Функция выбора цвета в зависимости от оценки
def get_color(score):
    if score >= 4:
        return '#ccffcc'   # Светло-зеленый
    elif score == 3:
        return '#ffffcc'  # Светло-желтый
    else:
        return '#ffcccc'  # Светло-коралловый


# Функция обработки данных и применения цвета
def process_scores(df, group_by='Упражнение'):

    if df.empty:
        return pd.DataFrame()

    numeric_cols = [ 'HEAD','TORSO','ARMS', 'LEGS', 'GENERAL']
    print('numeric_cols',numeric_cols)

    if group_by:
        results_df = df.groupby(group_by)[numeric_cols].mean().reset_index()
    else:
        results_df = pd.DataFrame([df[numeric_cols].mean()], index=[0])
        results_df['Упражнение'] = 'Нераспознано'

    results_df[numeric_cols] = results_df[numeric_cols].apply(lambda x: x.map(convert_score))

    print('results_df[numeric_cols] = results_df[numeric_cols].apply(lambda x: x.map(convert_score))',results_df)

    results_df['Количество'] = df.groupby(group_by).size().values if group_by else [len(df)]

    print("results_df['Количество']",results_df)
    
    return results_df

def get_workout_statistics(df):
    
    print('get_workout_statistics - начало работы')
    if df.empty:
        return pd.DataFrame(),0
    
    # Обработка данных
    filtered_df = df[df['GENERAL'] <= 0.95]
    if filtered_df.empty:
        results = pd.DataFrame()
    else:
        results = process_scores(filtered_df)

    # Считаем количество видов распознанных упражнений
    workout_types=len(results)

    print('filtered_df',filtered_df)
    print('results',results)

    # Собираем итоги по колонкам
    totals = {
        'Упражнение': 'ИТОГО',
        'HEAD': round(results['HEAD'].mean()),
        'TORSO': round(results['TORSO'].mean()),
        'ARMS': round(results['ARMS'].mean()),
        'LEGS': round(results['LEGS'].mean()),
        'GENERAL': round(results['GENERAL'].mean()),
        'Количество': round(results['Количество'].sum()) 
    }

    # display('filtered_df', filtered_df)
    # display('results',results)

    # Обработка "Нераспознанных" данных
    unrecognized_df = df[df['GENERAL'] > 0.95]
    # display('unrecognized_df',unrecognized_df)

    if not unrecognized_df.empty:
        unrecognized_scores = process_scores(unrecognized_df, group_by=None)
        # display('unrecognized_scores',unrecognized_scores)
        results = pd.concat([results, unrecognized_scores], ignore_index=True)
        totals['Количество']=results['Количество'].sum()

    # Создаем DataFrame из итогов
    totals_df = pd.DataFrame.from_dict(totals, orient='index').T
    # Подошьем итоги
    results = pd.concat([results, totals_df], axis=0, ignore_index=True)

    print('Объединеные results',results)    

    # Словарь для замены заголовков
    header_dict = {
        'Exercise': 'Упражнение',
        'HEAD': 'ГОЛОВА',
        'TORSO': 'ТОРС',
        'ARMS': 'РУКИ',
        'LEGS': 'НОГИ',
        'GENERAL': 'ОБЩЕЕ',
        'Bends_forvard': 'Наклоны вперед',
        'Squats': 'Приседания',
        'Количество': 'Количество'
        }

    # Переименование и замена значений
    results = results.replace(header_dict)
    print('Замена значений results',results)   

    # Переименование заголовков колонок
    results = results.rename(columns=header_dict)

    return results, workout_types 

# display('заменили значения results',results) 

# Функция для генерации HTML
def generate_html(df, time_spent, workout_types, daily_charge=4,overall_progress=37):
    
    score_cols = ['ГОЛОВА',	'ТОРС',	'РУКИ',	'НОГИ',	'ОБЩЕЕ']
    exercise_count = df['Количество'].iloc[-1]  # количество из последней строки
    average_score = df['ОБЩЕЕ'].iloc[-1]  # средняя оценка за тренировку

    html= """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
</head>
    <div style='border: 2px solid gray; padding: 10px;'>
    <h1 style='text-align: center;'>Поздравляем!</h1>
    <h2 style='text-align: center;'>Вы завершили тренировку</h2>
    <h3 style='text-align: center;'>Основные параметры:</h3>
"""
    html += f"<p style='text-align: center;'>Время: {time_spent} мин</p>"
    html += f"<p style='text-align: center;'>Видов упражнений: {workout_types}</p>" 
    html += "<h3 style='text-align: center;'>Таблица оценок за упражнения 2-5</h3>"
    html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    
    # Заголовки таблицы
    html += "<tr>"
    for col in df.columns:
        html += f"<th style='text-align: center; width: 50px; background-color: #A4C8E1; writing-mode: vertical-rl; text-orientation: sideways'>{col}</th>"
    html += "</tr>"
    
    for index, row in df.iterrows():
        html += "<tr>"
        for i,col in enumerate(df.columns):
            if index == len(df)-1:
                color= '#eeddff'
            elif col in score_cols:
                color = get_color(row[col])
            elif i == 0:
                color = '#f7f4ee'
            else:
                color = '#e6e6e6'
            html += f"<td style='background-color: {color}; text-align: center; padding: 5px;'>{row[col]}</td>"
        html += "</tr>"
    
    html += "</table>"
    
    # Добавляем информацию после таблицы
    html += f"<p style='text-align: center;'>Общая оценка за тренировку: <strong>{average_score:.1f}</strong></p>"
    html += f"<p style='text-align: center;'>Сегодня зарядился на: <strong>{daily_charge}%</strong></p>"
    html += f"<p style='text-align: center;'>Общий прогресс: <strong>{overall_progress}%</strong></p>"
    
    html += "</div>"
    
    return html

def generate_final_stats(html_path,stat_df,workout_time):
    
    # Превращаем данные анализа видео в агрегированную статистику
    results_df, workout_types = get_workout_statistics(stat_df)

    print('generate_final_stats: results_df', results_df)
    
    # формируем HTML для вывода в приложение
    if results_df.empty:
        html_output =   """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Нет упражнений</title>
    <style>
        body { display: flex;  justify-content: center; align-items: center;  height: 100vh; margin: 0; font-family: Arial, sans-serif; background-color: #f7f7f7; color: #333; }
        .container { text-align: center; }
        h1 {font-size: 24px; margin-bottom: 20px;}
        p { font-size: 18px; margin: 5px 0;}
    </style>
</head>
<body>
    <div class="container">
        <h1>В присланном видео не найдено упражнений :(</h1>
        <p>Попробуйте еще раз</p>
    </div>
</body>
</html>"""

    else:
        html_output = generate_html(results_df, workout_time, workout_types, daily_charge=4,overall_progress=37)
    
    print('html_output,html_path',html_output,html_path)

    try:
        with open(html_path, 'w', encoding='utf-8') as file:
            file.write(html_output)
        print(f"Файл успешно сохранен по пути: {html_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")

    print("HTML файл успешно сохранен.")

    # display(HTML(html_output))

    return results_df

# # тестовый блок
# html_path='output_html.html'

# # Инициализация и обработка данных
# stat_df = pd.DataFrame({
#     'HEAD': [1.289408, 0.544213, 0.760669, 1.088565, 1.032467, 0.604234, 0.666443, 0.799200, 1.072545, 1.072458, 1.133523],
#     'TORSO': [1.155901, 0.943849, 0.483235, 0.524183, 0.687679, 0.356416, 0.367577, 0.502435, 0.908116, 0.922537, 0.916569],
#     'ARMS': [1.406203, 1.214698, 0.602943, 0.613263, 0.731803, 0.706886, 0.703632, 0.799239, 1.234175, 1.135642, 1.181844],
#     'LEGS': [1.398884, 1.416506, 0.698345, 0.612926, 0.995476, 0.647110, 0.529414, 1.141296, 1.503557, 1.542323, 1.522648],
#     'GENERAL': [1.312599, 1.029817, 0.636298, 0.709734, 0.861856, 0.578662, 0.566766, 0.810543, 1.179598, 1.168240, 1.188646],
#     'Упражнение': ['Squats', 'Squats', 'Bends_forward', 'Bends_forward', 'Squats', 'Squats', 'Squats', 'Squats', 'Squats', 'Squats', 'Squats'],
#     'Вид': ['front'] * 11
# })

# workout_time=5
# generate_final_stats(html_path,stat_df,workout_time)

