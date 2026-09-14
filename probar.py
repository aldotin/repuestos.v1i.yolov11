from ultralytics import YOLO

model = YOLO("runs/detect/repuestos_yolo11/weights/best.pt")

results = model.predict(
    source="prueba/bujia1.jpg",
    device=0,
    conf=0.1,
    save=True
)

print("Prueba terminada.")

