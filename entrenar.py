from ultralytics import YOLO

# Cargar modelo YOLO11 nano preentrenado
model = YOLO("yolo11n.pt")

# Entrenar el modelo
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    name="repuestos_yolo11"
)

print("Entrenamiento terminado.")
