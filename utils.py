import os

def generate_filename(label, prefix, index):
    """
    สร้างชื่อไฟล์รูปพร้อม id
    เช่น:
    - แมว: A_1.jpg, A_2.jpg
    - หมา: B_1.jpg, B_2.jpg
    """
    return f"{prefix}_{index}.jpg"
def count_images(folder):
    count = 0
    for root, dirs, files in os.walk(folder):  # ✅ ใช้ os.walk เพื่ออ่าน subfolder ด้วย
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1
    return count
