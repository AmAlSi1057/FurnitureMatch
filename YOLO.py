import clip
import torch
from ultralytics import YOLO
import cv2
from PIL import Image

# טעינת המודלים
device = "cuda" if torch.cuda.is_available() else "cpu"
model_yolo = YOLO("yolo11n.pt")  # ניתן לשנות למודל גדול יותר
model_clip, preprocess = clip.load("ViT-B/32", device=device)

# קריאת תמונה
image_path = r"C:\Users\Aurora\source\repos\YOLO\YOLO\image_321.jpg"
image = cv2.imread(image_path)

# הרצת YOLO על התמונה
results = model_yolo(image)

# בחירת האובייקט הראשון שזוהה (אפשר לשנות)
for result in results:
    for box in result.boxes[:]:  # נבחר את הראשון ברשימה
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # קואורדינטות
        label = result.names[int(box.cls[0].item())]  # שם האובייקט

        # חיתוך התמונה לפי הקואורדינטות
        cropped_object = image[y1:y2, x1:x2]

        # המרת התמונה לפורמט שמתאים ל-CLIP
        pil_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))
        processed_image = preprocess(pil_image).unsqueeze(0).to(device)

        # המרת התמונה לווקטור בעזרת CLIP
        with torch.no_grad():
            image_features = model_clip.encode_image(processed_image)

        # הצגת האובייקט שנחתך
        cv2.imshow("Cropped Object", cropped_object)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # הדפסת הווקטור
        print(f"📌 וקטור עבור האובייקט {label}:")
        print(image_features.cpu().numpy())

        
vcvcccvcvc