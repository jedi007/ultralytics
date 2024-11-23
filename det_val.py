from ultralytics import YOLO

# Load a model
# model = YOLO("helmet_241009.pt")
model = YOLO("det_personup_helmet_241119.pt")

# Train the model
# train_results = model.train(
#     data="safety.yaml",  # path to dataset YAML
#     epochs=0,  # number of training epochs
#     imgsz=(384, 640),  # training image size
#     device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )

# Evaluate model performance on the validation set
# metrics = model.val()
model.val(data="safety.yaml", imgsz=(384, 640))