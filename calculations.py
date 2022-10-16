from numba import njit
from config import DISTANCE_COEF

LINE_WIDTH_MM = 1600
LINE_WIDTH = 915
FIRST_EDGE = 562
SECOND_EDGE = 404
THIRD_EDGE = 351


@njit
def count_size(box):
    x_left = int(box[0])
    y_left = int(box[1])
    x_right = int(box[2])
    y_right = int(box[3])

    width = y_right - y_left
    length = x_right - x_left

    # Приблизительный перевод в миллиметры
    width_mm = width * LINE_WIDTH_MM / LINE_WIDTH
    length_mm = length * LINE_WIDTH_MM / LINE_WIDTH
    # Увеличение в случае удаленности куска от камеры
    # Условные границы в пикселях начиная с низу изображения
    if y_right < FIRST_EDGE:
        return max(width_mm, length_mm)
    elif FIRST_EDGE < y_right < SECOND_EDGE:
        return max(width_mm, length_mm) * DISTANCE_COEF
    elif SECOND_EDGE < y_right < THIRD_EDGE:
        return max(width_mm, length_mm) * (DISTANCE_COEF ** 2)
    return 0
