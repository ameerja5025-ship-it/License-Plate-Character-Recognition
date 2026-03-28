import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import Levenshtein
import os

class ALPRSystem:
    def __init__(self, yolo_model_path):
        print("جاري تحميل نموذج YOLO...")
        self.detector = YOLO(yolo_model_path)
        
        print("جاري تحميل EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False) 

    def preprocess_for_ocr(self, cropped_plate):
        """ معالجة الصورة لحل مشاكل تباين الإضاءة والظلال """
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def process_image(self, image_path):
        """ تحديد موقع اللوحة، استخراج النص، ورسم النتائج على الصورة """
        img = cv2.imread(image_path)
        if img is None:
            return "", None

        annotated_img = img.copy()
        results = self.detector(img, verbose=False)
        best_text = ""
        highest_confidence = 0.0
        best_box = None 

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue
                
                processed_plate = self.preprocess_for_ocr(plate_crop)
                ocr_results = self.reader.readtext(processed_plate, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                
                for (bbox, text, prob) in ocr_results:
                    clean_text = ''.join(e for e in text if e.isalnum()).upper()
                    if prob > highest_confidence and len(clean_text) >= 3:
                        highest_confidence = prob
                        best_text = clean_text
                        best_box = (x1, y1, x2, y2)
                        
        if best_box and best_text:
            bx1, by1, bx2, by2 = best_box
            # رسم المربع الأخضر وكتابة النص
            cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            # التأكد من أن النص لا يخرج خارج إطار الصورة من الأعلى
            text_y = max(by1 - 15, 20) 
            cv2.putText(annotated_img, best_text, (bx1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                        
        return best_text, annotated_img

def calculate_character_accuracy(ground_truths, predictions):
    """ حساب الدقة على مستوى الحرف لمتطلبات الورقة البحثية """
    total_characters = 0
    total_errors = 0
    for gt, pred in zip(ground_truths, predictions):
        total_characters += len(gt)
        errors = Levenshtein.distance(gt, pred)
        total_errors += errors
    if total_characters == 0: return 0.0
    accuracy = max(0, (total_characters - total_errors) / total_characters) * 100
    return accuracy

# ==========================================
# وحدة التنفيذ الرئيسية
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = 'yolov8n.pt' 
    alpr = ALPRSystem(MODEL_PATH)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
    
    # =======================================================
    # قم بتعديل هذا القاموس ليحتوي على أسماء صورك الـ 20 والنص الحقيقي لها
    # =======================================================
    test_dataset = {
        "car1.jpg": "HAMMAS",
        "car2.jpg": "ABC1234",
        "car3.jpg": "XYZ987",
    }
    
    ground_truths = []
    predictions = []
    
    print("\n" + "="*40)
    print("بدء عملية الفحص والتعرف على اللوحات...")
    print("="*40)
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"\n[خطأ]: يرجى التأكد من إنشاء مجلد باسم 'test_images' في هذا المسار:\n{BASE_DIR}")
    else:
        for file_name, true_text in test_dataset.items():
            img_path = os.path.join(TEST_IMAGES_DIR, file_name)
            
            if os.path.exists(img_path):
                print(f"\n>>> جاري فحص الصورة: {file_name} ...")
                
                predicted_plate, result_img = alpr.process_image(img_path)
                
                ground_truths.append(true_text)
                predictions.append(predicted_plate)
                
                print(f"النص الحقيقي المكتوب: {true_text}")
                print(f"النص المستخرج آلياً: {predicted_plate if predicted_plate else '[لم يتم التعرف]'}")
                
                if result_img is not None:
                    # تصغير الصورة لتناسب شاشة العرض
                    display_img = cv2.resize(result_img, (800, 600))
                    cv2.imshow(f"Result - {file_name} (Press ANY KEY to continue)", display_img)
                    cv2.waitKey(0) # إيقاف مؤقت حتى تضغط أي زر للانتقال للصورة التالية
            else:
                print(f"\n[تحذير]: الصورة {file_name} غير موجودة في مجلد test_images.")
        
        cv2.destroyAllWindows()
        
        if ground_truths:
            char_accuracy = calculate_character_accuracy(ground_truths, predictions)
            print("\n" + "="*40)
            print("--- النتائج النهائية للتقييم ---")
            print(f"إجمالي الصور المعالجة: {len(ground_truths)}")
            print(f"الدقة على مستوى الحرف (Accuracy): {char_accuracy:.2f}%")
            print("="*40)