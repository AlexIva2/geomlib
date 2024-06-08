# geomlib
# Описание библиотеки geomlib
Библиотека geomlib предназначена для работы с геометрическими объектами в трехмерном пространстве, такими как прямые и плоскости. Она предоставляет инструменты для определения взаимного расположения прямой и плоскости, взаимного расположения трех плоскостей, а также взаимного расположения двух прямых.

## Основные возможности библиотеки включают:

### Форматирование уравнений прямых и плоскостей в различных формах (общие, параметрические, канонические уравнения);

-Определение типов взаимного расположения двух прямых (пересекающиеся, параллельные, скрещивающиеся, совпадающие);

-Определение типов взаимного расположения прямой и плсокости (пересекающиеся, параллельные, принадлежности прямой плоскости);

-Определение типов взаимного расположения трёх плоскостей:

1.Три данные плоскости имеют и притом только одну общую точку;/
2.Плоскости попарно пересекаются, причем прямые пересечения попарно различны;
3.Две плоскости параллельны, а третья их пересекает;
4.Плоскости попарно различны и проходят через одну прямую;
5.Две плоскости совпадают, а третья их пересекает;
6.Плоскости попарно параллельны;
7.Две плоскости совпадают, а третья им параллельна;
8.Все плоскости совпадают.

-Генерация различных типов уравнений с разными типами взаимного расположения.

# Инструкция по применению библиотеки geomlib
## Установка и импорт
1. Склонируйте репозиторий:
   `git clone https://github.com/AlexIva2/geomlib.git`
2. Перейдите в директорию geomlib:
   `cd geomlib`
3. Импортируйте необходимые функции из utils.py:
   `from utils import *`  # Импортировать все функции
## Пример использывания
`print(GeometryGenerator.generate_variants(10, txt=True))` # Генерация различных типов уравнений прямой и плсокости и вывод 10 случаев взаимного расположения прямой и плоскоскости.
