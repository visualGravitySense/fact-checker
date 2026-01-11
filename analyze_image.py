"""
Детальный анализ изображения на реальность/подделку
"""

import sys
import io
# Исправление кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from face_real_fake_detector import FaceRealFakeDetector
import cv2
import numpy as np

# Создаем детектор
detector = FaceRealFakeDetector()

# Анализируем изображение
image_path = "613648953_1449572173841888_2396410150948109536_n.jpg"

print("="*60)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ИЗОБРАЖЕНИЯ")
print("="*60)
print(f"Изображение: {image_path}\n")

# Загружаем изображение
image = cv2.imread(image_path)
if image is None:
    print("Ошибка: не удалось загрузить изображение")
    exit(1)

print(f"Размер изображения: {image.shape[1]}x{image.shape[0]} пикселей\n")

# Пробуем с детекцией лица
print("1. Анализ с детекцией лица:")
print("-" * 60)
result_with_face = detector.analyze_image(image_path, use_face_detection=True)
print(f"Результат: {'РЕАЛЬНОЕ' if result_with_face['is_real'] else 'ПОДДЕЛЬНОЕ'}")
print(f"Уверенность: {result_with_face['confidence']:.2%}")

# Анализ всего изображения
print("\n2. Анализ всего изображения:")
print("-" * 60)
result_full = detector.analyze_image(image_path, use_face_detection=False)
print(f"Результат: {'РЕАЛЬНОЕ' if result_full['is_real'] else 'ПОДДЕЛЬНОЕ'}")
print(f"Уверенность: {result_full['confidence']:.2%}")

# Детальная информация о ковариационной матрице
if result_full.get('covariance_matrix'):
    C = np.array(result_full['covariance_matrix'])
    print("\n3. Ковариационная матрица градиентов:")
    print("-" * 60)
    print(f"  C = [{C[0,0]:.2f}  {C[0,1]:.2f}]")
    print(f"      [{C[1,0]:.2f}  {C[1,1]:.2f}]")
    
    eigenvals = np.linalg.eigvals(C)
    eigenvals = np.sort(eigenvals)[::-1]
    print(f"\n  Собственные значения:")
    print(f"    λ₁ = {eigenvals[0]:.2f} (наибольшее)")
    print(f"    λ₂ = {eigenvals[1]:.2f} (наименьшее)")
    
    # Число обусловленности
    if eigenvals[1] > 0:
        condition_num = eigenvals[0] / eigenvals[1]
        print(f"\n  Число обусловленности: {condition_num:.2f}")
        print(f"    (отношение λ₁/λ₂)")
        if condition_num > 2.0:
            print(f"    → Высокое число обусловленности указывает на структурированные градиенты")
            print(f"    → Это характерно для РЕАЛЬНЫХ изображений")
        else:
            print(f"    → Низкое число обусловленности может указывать на менее структурированные градиенты")
    
    # След и определитель
    trace = np.trace(C)
    det = np.linalg.det(C)
    print(f"\n  След матрицы (trace): {trace:.2f}")
    print(f"  Определитель (det): {det:.2f}")

# Анализ признаков
if result_full.get('features'):
    features = result_full['features']
    print("\n4. Извлеченные признаки:")
    print("-" * 60)
    print(f"  Собственные значения: [{features[0]:.2f}, {features[1]:.2f}]")
    print(f"  След: {features[2]:.2f}")
    print(f"  Определитель: {features[3]:.2f}")
    print(f"  Число обусловленности: {features[4]:.2f}")
    print(f"  Среднее |Gx|: {features[5]:.2f}")
    print(f"  Среднее |Gy|: {features[6]:.2f}")
    print(f"  Стандартное отклонение Gx: {features[7]:.2f}")
    print(f"  Стандартное отклонение Gy: {features[8]:.2f}")
    print(f"  Корреляция Gx-Gy: {features[9]:.2f}")

print("\n" + "="*60)
print("ВЫВОД:")
print("="*60)
if result_full['is_real']:
    print("Изображение классифицировано как РЕАЛЬНОЕ")
    print("Градиенты показывают структурированные паттерны,")
    print("характерные для физических объектов и реального освещения.")
else:
    print("Изображение классифицировано как ПОДДЕЛЬНОЕ")
    print("Градиенты показывают менее структурированные паттерны,")
    print("что может указывать на AI-генерацию или манипуляцию.")

if result_full.get('note'):
    print(f"\nПримечание: {result_full['note']}")
print("="*60)
