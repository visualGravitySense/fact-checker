"""
Визуализация градиентов изображения для анализа реальности/подделки
"""

import sys
import io
# Исправление кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from face_real_fake_detector import FaceRealFakeDetector

def visualize_gradients(image_path, output_prefix="gradient_analysis"):
    """
    Создает визуализацию градиентов изображения.
    
    Args:
        image_path: Путь к изображению
        output_prefix: Префикс для выходных файлов
    """
    print("Загрузка изображения...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    
    detector = FaceRealFakeDetector()
    
    # Пробуем найти лицо
    faces = detector.detect_face(image)
    if len(faces) > 0:
        print(f"Обнаружено лиц: {len(faces)}")
        # Используем самое большое лицо
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(min(w, h) * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        roi = image[y:y+h, x:x+w]
        print(f"Область лица: {w}x{h} пикселей")
    else:
        print("Лицо не обнаружено, анализируем все изображение")
        roi = image
    
    # Конвертируем в яркость
    print("Вычисление яркости...")
    luminance = detector.rgb_to_luminance(roi)
    
    # Вычисляем градиенты
    print("Вычисление градиентов...")
    Gx, Gy = detector.compute_gradients(luminance)
    
    # Вычисляем величину градиента
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # Нормализуем для визуализации
    def normalize_for_display(img):
        """Нормализует изображение для отображения (0-255)"""
        img_abs = np.abs(img)
        if img_abs.max() > 0:
            normalized = (img_abs / img_abs.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img, dtype=np.uint8)
        return normalized
    
    Gx_display = normalize_for_display(Gx)
    Gy_display = normalize_for_display(Gy)
    magnitude_display = normalize_for_display(gradient_magnitude)
    
    # Создаем фигуру с несколькими подграфиками
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Оригинальное изображение
    ax1 = plt.subplot(2, 3, 1)
    if len(roi.shape) == 3:
        display_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else:
        display_img = roi
    ax1.imshow(display_img)
    ax1.set_title('Оригинальное изображение', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Яркость (luminance)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(luminance, cmap='gray')
    ax2.set_title('Яркость (Luminance)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Горизонтальный градиент (Gx)
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(Gx_display, cmap='gray')
    ax3.set_title('Горизонтальный градиент (Gx = ∂L/∂x)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Вертикальный градиент (Gy)
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(Gy_display, cmap='gray')
    ax4.set_title('Вертикальный градиент (Gy = ∂L/∂y)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 5. Величина градиента
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(magnitude_display, cmap='hot')
    ax5.set_title('Величина градиента (|∇L|)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # 6. Комбинированная визуализация (цветное кодирование направления)
    ax6 = plt.subplot(2, 3, 6)
    # Вычисляем направление градиента
    gradient_direction = np.arctan2(Gy, Gx)
    # Нормализуем направление для визуализации
    direction_normalized = (gradient_direction + np.pi) / (2 * np.pi)  # 0-1
    magnitude_normalized = gradient_magnitude / gradient_magnitude.max() if gradient_magnitude.max() > 0 else gradient_magnitude
    
    # Создаем HSV изображение: Hue = направление, Value = величина
    hsv = np.zeros((*gradient_direction.shape, 3))
    hsv[:, :, 0] = direction_normalized  # Hue (направление)
    hsv[:, :, 1] = 1.0  # Saturation
    hsv[:, :, 2] = magnitude_normalized  # Value (величина)
    
    rgb_direction = cm.hsv(hsv[:, :, 0])[:, :, :3]
    rgb_direction = rgb_direction * magnitude_normalized[:, :, np.newaxis]
    
    ax6.imshow(rgb_direction)
    ax6.set_title('Направление градиента (цвет)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Сохраняем фигуру
    output_file = f"{output_prefix}_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nВизуализация сохранена: {output_file}")
    
    # Создаем отдельные файлы для каждого градиента
    cv2.imwrite(f"{output_prefix}_luminance.png", luminance.astype(np.uint8))
    cv2.imwrite(f"{output_prefix}_gradient_x.png", Gx_display)
    cv2.imwrite(f"{output_prefix}_gradient_y.png", Gy_display)
    cv2.imwrite(f"{output_prefix}_gradient_magnitude.png", magnitude_display)
    
    print(f"Отдельные изображения сохранены:")
    print(f"  - {output_prefix}_luminance.png")
    print(f"  - {output_prefix}_gradient_x.png")
    print(f"  - {output_prefix}_gradient_y.png")
    print(f"  - {output_prefix}_gradient_magnitude.png")
    
    # Статистика
    print("\n" + "="*60)
    print("СТАТИСТИКА ГРАДИЕНТОВ:")
    print("="*60)
    print(f"Горизонтальный градиент (Gx):")
    print(f"  Среднее: {np.mean(Gx):.2f}")
    print(f"  Стандартное отклонение: {np.std(Gx):.2f}")
    print(f"  Мин/Макс: {np.min(Gx):.2f} / {np.max(Gx):.2f}")
    print(f"\nВертикальный градиент (Gy):")
    print(f"  Среднее: {np.mean(Gy):.2f}")
    print(f"  Стандартное отклонение: {np.std(Gy):.2f}")
    print(f"  Мин/Макс: {np.min(Gy):.2f} / {np.max(Gy):.2f}")
    print(f"\nВеличина градиента:")
    print(f"  Среднее: {np.mean(gradient_magnitude):.2f}")
    print(f"  Стандартное отклонение: {np.std(gradient_magnitude):.2f}")
    print(f"  Мин/Макс: {np.min(gradient_magnitude):.2f} / {np.max(gradient_magnitude):.2f}")
    
    # Вычисляем ковариационную матрицу для дополнительной информации
    C, features = detector.compute_covariance_matrix(Gx, Gy)
    if C is not None:
        print(f"\nКовариационная матрица:")
        print(f"  [{C[0,0]:.2f}  {C[0,1]:.2f}]")
        print(f"  [{C[1,0]:.2f}  {C[1,1]:.2f}]")
        eigenvals = np.linalg.eigvals(C)
        eigenvals = np.sort(eigenvals)[::-1]
        print(f"  Собственные значения: [{eigenvals[0]:.2f}, {eigenvals[1]:.2f}]")
    
    plt.show()
    
    return fig

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Визуализация градиентов изображения')
    parser.add_argument('image_path', type=str, help='Путь к изображению')
    parser.add_argument('--output', type=str, default='gradient_analysis', 
                       help='Префикс для выходных файлов')
    
    args = parser.parse_args()
    
    visualize_gradients(args.image_path, args.output)
