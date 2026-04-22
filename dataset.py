# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("daudaudinang/ui-elements-detection-dataset")

# print("Path to dataset files:", path)

from ultralytics import YOLO

# def main():
#     model = YOLO("yolov8n.pt")

# model.train(
#     data=r"C:\Users\Raunak\Documents\Kai\dataset.yaml",
#     epochs=30,
#     imgsz=640,
#     batch=3,
#     device=0,
# )


# if __name__ == "__main__":
#     main()

model = YOLO(r"C:\Users\Raunak\Documents\Kai\runs\detect\train-14\weights\best.pt")

results = model("Screenshot 2026-04-22 155928.png", show=True)
results[0].save(filename="output.png")
