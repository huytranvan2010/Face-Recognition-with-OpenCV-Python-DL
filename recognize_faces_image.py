# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image 1.png
import face_recognition
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
# đường dẫn đến file encodings đã lưu
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True, help="path to the test image")
# nếu chạy trên CPU hay embedding devices thì để hog, còn khi tạo encoding vẫn dùng cnn cho chính xác
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face dettection model to use: cnn or hog")
args = vars(ap.parse_args())

# load the known faces and encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())      # loads - load từ file

# load image và chuyển từ BGR to RGB (dlib cần để chuyển về encoding)
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# CŨng làm tương tự cho ảnh test, detect face, extract face ROI, chuyển về encoding
# rồi cuối cùng là so sánh kNN để recognize
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# khởi tạo list chứa tên các khuôn mặt phát hiện được
# nên nhớ trong 1 ảnh có thể phát hiện được nhiều khuôn mặt nhé
names = []

# duyệt qua các encodings của faces phát hiện được trong ảnh
for encoding in encodings:
    # khớp encoding của từng face phát hiện được với known encodings (từ datatset)
    # so sánh list of known encodings và encoding cần check, sẽ trả về list of True/False xem từng known encoding có khớp với encoding check không
    # có bao nhiêu known encodings thì trả về list có bấy nhiêu phần tử
    # trong hàm compare_faces sẽ tính Euclidean distance và so sánh với tolerance=0.6 (mặc định), nhó hơn thì khớp, ngược lại thì ko khớp (khác người)
    matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)      # có thể điều chỉnh tham số cuối
    name = "Unknown"    # tạm thời vậy, sau này khớp thì đổi tên

    # Kiểm tra xem từng encoding có khớp với known encodings nào không,
    if True in matches:
        # lưu các chỉ số mà encoding khớp với known encodings (nghĩa là b == True)
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]

        # tạo dictionary để đếm tổng số lần mỗi face khớp
        counts = {}
        # duyệt qua các chỉ số được khớp và đếm số lượng 
        for i in matchedIdxs:
            name = data["names"][i]     # tên tương ứng known encoding khiowps với encoding check
            counts[name] = counts.get(name, 0) + 1  # nếu chưa có trong dict thì + 1, có rồi thì lấy số cũ + 1

        # lấy tên có nhiều counts nhất (tên có encoding khớp nhiều nhất với encoding cần check)
        # có nhiều cách để có thể sắp xếp list theo value ví dụ new_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        # nó sẽ trả về list of tuple, mình chỉ cần lấy name = new_dic[0][0]
        name = max(counts, key=counts.get)

    names.append(name)

# Duyệt qua các bounding boxes và vẽ nó trên ảnh kèm thông tin
# Nên nhớ recognition_face trả bounding boxes ở dạng (top, rights, bottom, left)
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15

    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

cv2.imshow("Image", image)
cv2.waitKey(0)




