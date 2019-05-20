import sys

import cv2
import numpy as np

from object_tracking.object_tracking import ObjectTracker
from ssd.mobilenet_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from state_determining.car_state import StateQualifier
from state_determining.light_state import determine_light_on
from utils.misc import Timer

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
state_dict = None
parking_place_box = [634, 358, 817, 728]
state_qualifier = StateQualifier(parking_place_box)
car_tracker = ObjectTracker(class_names)

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    if determine_light_on(image, 20):
        timer.start()
        prediction = predictor.predict(image, 10, 0.2)
        if prediction[0].size(0) > 0:
            cars, other, ids = car_tracker.track('car', prediction)

            if car_tracker.prev_values is not None:
                state_dict = state_qualifier.get_state_dict(cars[0], ids, car_tracker.prev_values[0],
                                                            car_tracker.prev_values[1])
            prev_vals = cars[0], ids
            boxes = np.vstack((cars[0], other[0]))
            labels = np.concatenate((cars[1], other[1]))
            probabilities = np.concatenate((cars[2], other[2]))

        interval = timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.shape[0]))
        for i in range(boxes.shape[0]):
            box = boxes[i, :].astype(int)
            color = (255, 255, 0)
            if i < ids.shape[0] and state_dict is not None:
                label = f"{class_names[labels[i]]}: {probabilities[i]:.2f} {state_dict[ids[i]]}"
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
