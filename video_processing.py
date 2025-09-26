import cv2 as cv
import numpy as np
import sys

# Диапазоны красного цвета в HSV (две части спектра)
HSV_RED_LOWER_1 = np.array([0, 100, 100])
HSV_RED_UPPER_1 = np.array([10, 255, 255])
HSV_RED_LOWER_2 = np.array([170, 100, 100])
HSV_RED_UPPER_2 = np.array([180, 255, 255])

def create_red_mask(frame):
    """
    Создает бинарную маску красного цвета для кадра.

    Аргумент:
        frame (np.ndarray): Цветной кадр BGR.

    Возвращает:
        mask (np.ndarray): Бинарная маска красного цвета (0 или 255).
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, HSV_RED_LOWER_1, HSV_RED_UPPER_1)
    mask2 = cv.inRange(hsv, HSV_RED_LOWER_2, HSV_RED_UPPER_2)
    mask = cv.bitwise_or(mask1, mask2)
    return mask

def draw_rectangles(frame, mask, min_area = 100):
    """
    Находит контуры на маске и рисует зеленые прямоугольники вокруг объектов.

    Аргумент:
        frame (np.ndarray): Исходный цветной кадр.
        mask (np.ndarray): Бинарная маска объектов.
        min_area (int): Минимальная площадь контура для фильтрации шума.

    Возвращает:
        np.ndarray: Кадр с нарисованными прямоугольниками.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) > min_area:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def process_video(input_path, output_path):
    """
    Обрабатывает видео: выделяет красные объекты, рисует вокруг них прямоугольники,
    показывает результат и опционально сохраняет в новый файл.

    Аргумент:
        input_path (str): Путь к исходному видеофайлу.
        output_path (str, optional): Путь для сохранения результата. Если None, не сохраняет.

    Возвращает:
        None
    """
    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {input_path}")

    # Получаем параметры видео
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv.CAP_PROP_FPS)) or 30

     # Проверка корректности параметров
    if width <= 0 or height <= 0:
        raise ValueError("Некорректные размеры кадра видео.")

    # Настройка записи видео
    out = None
    if output_path:
        out = cv.VideoWriter(
            output_path,
            cv.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        if not out.isOpened():
            raise IOError(f"Не удалось создать файл для записи видео: {output_path}")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Видео закончилось или не удалось прочитать кадр.")
            break

        mask = create_red_mask(frame)
        frame_with_rect = draw_rectangles(frame, mask)

        # Показываем результат
        cv.imshow("Original + Rectangles", frame_with_rect)
        cv.imshow("Red Mask", mask)

        # Сохраняем в файл с проверкой
        if out:
            try:
                out.write(frame_with_rect)
            except Exception as e:
                print(f"Ошибка записи кадра в видео: {e}")

        key = cv.waitKey(1) & 0xFF
        if key in [27, ord('q')]:# ESC или q
            break

    video.release()
    if out:
        out.release()
    cv.destroyAllWindows()

def main():
    """
    Главная функция запуска кода.
    """
    input_video = "original.mp4"
    output_video = "output.mp4"

    try:
        process_video(input_video, output_video)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка видеофайла: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"Ошибка записи видео: {e}")
        sys.exit(1)
    except Exception as e:
        print("Произошла непредвиденная ошибка:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
