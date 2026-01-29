from ultralistics import YOLO
import cv2

# загружаем модель yolo скелет человека
# создаём переменную для хранения модели 
model = YOLO('yolo8n-pose.pt')

#переменна для запуска камеры подключенной к пк или втроенной ноута
cap = cv2.VideoCapture()
while True:
    # захват кадра
    ret.frame = cap.read()
    if not ret:
        break
    #детекция позы человека
    result = model(frame,verbose=False[0])

    # Рисуем результат
    annotated_frame = result.plot()