import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

#Config value
video_path = "data_ext/highway.mp4"
conf_threshold = 0.5 #tren 0.5 => chap nhan ket qua predict
tracking_class = 2 #id cua car trong coco dataset, neu muon tracking all => thay 2 = None

#khoi tao deepsort
tracker = DeepSort(max_age=5) #sau 5 lan ma ko thay vat the => xoa vat the khoi bo nho

#khoi tao yolov9
device = "mps:0"  #GPU => "cuda", CPU => "cpu"
model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
model = AutoShape(model)

#load classname tu file classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size = (len(class_names), 3))
tracks = []

#khoi tao video capture de doc tu file video:
cap = cv2.VideoCapture(video_path)

#doc tung frame tu video:
while True:
    #doc
    ret, frame = cap.read()

    if not ret:
        continue

    #dua qua model de detect
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4], #bbox tra ra toa do
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None: #all
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue 
        
        detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])
            
    #cap nhat, gan id = deepsort
    tracks = tracker.update_tracks(detect, frame=frame)
    #ve len man hinh cac khung chu nhat kem id
    for track in tracks:
        if track.is_confirmed(): #duoc xac nhan boi tracker moi lam
            track_id = track.track_id

            #Lay toa do, class_id de ve len hinh anh
            ltrb = track.to_ltrb() 
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1-1, y1-20), (x1+len(label)*12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #show
    cv2.imshow("OT", frame)
    #q to quit
    if cv2.waitKey(1) == ord("q"):
        break 

cap.release()
cv2.destroyAllWindows()
