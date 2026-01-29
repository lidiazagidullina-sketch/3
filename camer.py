from ultralytics import YOLO
import cv2

# загружаем модель yolo скелет человека
# создаём переменную для хранения модели 
model = YOLO('yolov8n-pose.pt')

#переменна для запуска камеры подключенной к пк или втроенной ноута
cap = cv2.VideoCapture(0)
while True:
    # захват кадра
    ret,frame = cap.read()
    if not ret:
        break
    #детекция позы человека
    result = model(frame,verbose=False)[0]

    # Рисуем результат
    annotated_frame = result.plot()

    #Создаём скелет поверх отрисовки кадра человека с захватом колизии 
    cv2.addWeighted(frame,0.3, annotated_frame)


     #выведем небольшую информацию на экран 
    cv2.putText(annotated_frame,f"Общее количество людей: {len(result.keypoints)}",
    (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2) 

    # показать на экране
    cv2.imshow("Детектор",annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

 #запуск и разветка
cap.release()
cv2.destroyAllWindows()

