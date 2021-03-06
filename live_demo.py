import sys

import cv2
import torch

from car_system import CarSystem
from data_processing.ocr_data import CharDataset
from lpr.LPR import LPR
from lpr.character_segmentation import CharSeg
from lpr.cnn import CNN
from lpr.license_plate_detector import LicensePlateDetector
from object_tracking.object_tracking import ObjectTracker
from ssd.mobilenet_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from ssd.utils.misc import Timer
from state_determining.car_state import StateQualifier

if len(sys.argv) < 3:
    print('Usage: python live_demo.py <model path> <label path> [video file]')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]

if len(sys.argv) >= 4:
    cap = cv2.VideoCapture(sys.argv[3])  # capture from file
else:
    cap = cv2.VideoCapture(0)  # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

timer = Timer()
parking_place_box = [634, 358, 817, 728]
state_qualifier = StateQualifier(parking_place_box)
car_tracker = ObjectTracker(class_names)
plate_detector = LicensePlateDetector((20, 100, 100, 500))
dataset = CharDataset(root='dataset', label_file='labels.txt', multiply=0)
s_dict = torch.load('models/cnn.ckpt', map_location=lambda storage, loc: storage)
cnn = CNN(num_classes=36)
cnn.load_state_dict(s_dict)
char_seg: CharSeg = CharSeg((6, 80, 6, 60), padding=0)
lpr = LPR(char_seg, plate_detector, cnn, dataset)

frame_skip = 0

car_system = CarSystem(predictor, state_qualifier, car_tracker,
                       lpr, frame_skip, 20, 0.3)
lights = False
frame_n = 0

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    ids, boxes, labels, probabilities, state_dict, plates, light_on = car_system.handle_frame(image)
    interval = timer.end()
    if light_on and not lights:
        lights = True
    elif not light_on and lights:
        lights = False
    print("Frame:", frame_n)
    print("LIGHTS", "ON" if light_on else "OFF")
    if lights:
        print("Found", boxes.shape[0], "objects,", ids.shape[0], "of them are cars")

        for i in range(0, boxes.shape[0]):
            if i < ids.shape[0]:
                if state_dict[ids[i]] == "ARRIVED":
                    print('Car ID:', ids[i], "box:", boxes[i], "state:", state_dict[ids[i]], "plate:", plates)
                else:
                    print('Car ID:', ids[i], "box:", boxes[i], "state:", state_dict[ids[i]])
            else:
                print('Object box:', boxes[i])
    for i in range(boxes.shape[0]):
        box = boxes[i, :].astype(int)
        color = (255, 255, 0)
        if i < ids.shape[0] and state_dict is not None:
            label = f"{class_names[labels[i]]}: {probabilities[i]:.2f} {state_dict[ids[i]]}"
            if state_dict[ids[i]] == 'MOVE':
                color = (255, 255, 0)
            if state_dict[ids[i]] == 'LEFT':
                color = (0, 0, 255)
            if state_dict[ids[i]] == 'ARRIVED':
                color = (0, 255, 40)
        else:
            label = f"{class_names[labels[i]]}: {probabilities[i]:.2f}"

        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)

        cv2.putText(orig_image, label,
                    (box[0] + 6, box[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA)
    cv2.imshow('annotated', orig_image)
    frame_n = frame_n + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
