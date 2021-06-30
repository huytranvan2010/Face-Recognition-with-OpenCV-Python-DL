from os import write
import face_recognition
import argparse
import pickle
import cv2
import time
import imutils

ap = argparse.ArgumentParser()
# đường dẫn đến file encodings đã lưu
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
# nếu muốn lưu video từ webcam
ap.add_argument("-o", "--output", type=str, help="path to the output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
# nếu chạy trên CPU hay embedding devices thì để hog, còn khi tạo encoding vẫn dùng cnn cho chính xác
# ko có GPU thì nên để hog thôi nhé, cứ thử xem sao
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face dettection model to use: cnn or hog")
args = vars(ap.parse_args())

# load the known faces and encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb"))      # loads - load từ file

# Khởi tạo video stream và pointer to the output video file, để camera warm up một chút
print("[INFO] starting video stream...")
video = cv2.VideoCapture(0)     # có thể chọn cam bằng cách thay đổi src
writer = None
time.sleep(2.)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # chuyển frame từ BGR to RGB, resize để tăng tốc độ xử lý
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750)
    # hệ số scale từ ảnh gốc (frame) về rgb, tí phải dùng bên dưới
    r = frame.shape[1] / float(rgb.shape[1])

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
        matches = face_recognition.compare_faces(data["encodings"], encoding)
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
        """ Do đang làm việc với rgb đã resize rồi nên cần rescale về ảnh gốc (frame), nhớ chuyển về int """
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15

        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    """ Nếu writer=None và mình muốn lưu video thì lưu vào witer"""
    if writer == None and args["output"] is not None:   # nếu ko truyền vào nó cho là None
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)    # 20 à frames

    if writer is not None:  # tiếp tục ghi frame đã chèn bboxes, name vào
        writer.write(frame)

    """ Check xem có muốn hiển thị ra màn hình ko, mặc định là có"""
    if args["display"] > 0:
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 

video.release()
cv2.destroyAllWindows() 

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()




