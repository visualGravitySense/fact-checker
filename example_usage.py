"""
Пример использования FaceRealFakeDetector
"""

from face_real_fake_detector import FaceRealFakeDetector
import cv2

# Создаем детектор
detector = FaceRealFakeDetector()

# Анализируем изображение
image_path = "613648953_1449572173841888_2396410150948109536_n.jpg"

print("Анализ изображения...")
result = detector.analyze_image(image_path, use_face_detection=True)

# Выводим результаты
print("\nРезультаты анализа:")
print(f"Изображение: {image_path}")
print(f"Результат: {'РЕАЛЬНОЕ' if result['is_real'] else 'ПОДДЕЛЬНОЕ'}")
print(f"Уверенность: {result['confidence']:.2%}")

if result.get('eigenvalues'):
    print(f"\nСобственные значения ковариационной матрицы:")
    for i, val in enumerate(result['eigenvalues'], 1):
        print(f"  λ{i} = {val:.2f}")

if result.get('covariance_matrix'):
    print(f"\nКовариационная матрица:")
    C = result['covariance_matrix']
    print(f"  [{C[0][0]:.2f}  {C[0][1]:.2f}]")
    print(f"  [{C[1][0]:.2f}  {C[1][1]:.2f}]")

# Визуализация (опционально)
image = cv2.imread(image_path)
if image is not None:
    # Показываем изображение с результатом
    faces = detector.detect_face(image)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if result['is_real'] else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            label = "REAL" if result['is_real'] else "FAKE"
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Сохраняем результат
    output_path = "result_" + image_path
    cv2.imwrite(output_path, image)
    print(f"\nРезультат сохранен в: {output_path}")
