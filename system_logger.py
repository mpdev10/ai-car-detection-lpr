import argparse

import torch
from cv2 import cv2

from car_system import CarSystem
from data_processing.ocr_data import CharDataset
from lpr.LPR import LPR
from lpr.character_segmentation import CharSeg
from lpr.cnn import CNN
from lpr.license_plate_detector import LicensePlateDetector
from object_tracking.object_tracking import ObjectTracker
from ssd.mobilenet_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from state_determining.car_state import StateQualifier

parser = argparse.ArgumentParser(
    description='Text Logger Script'
)

parser.add_argument('--license_plate')

parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--ssd', default='models/SSD-Model.pth')
parser.add_argument('--ssd_labels', default='models/labels.txt')
parser.add_argument('--ssd_prob_th', default=0.3)
parser.add_argument('--video_file', default=None)
parser.add_argument('--parking_box', nargs='+', type=int, default=[634, 358, 817, 728])
parser.add_argument('--plate_size', nargs='+', type=int, default=[20, 100, 100, 500])
parser.add_argument('--cnn_dataset_root', default='dataset')
parser.add_argument('--cnn_labels', default='labels.txt')
parser.add_argument('--cnn', default='models/cnn.ckpt')
parser.add_argument('-char_properties', nargs='+', type=int, default=[15, 40, 7, 20, 1])
parser.add_argument('--frame_skip', default=0)
parser.add_argument('--light_th', default=20)

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Using CUDA.")

if __name__ == '__main__':

    if args.video_file:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1080)

    print("Loading SSD")
    ssd_class_names = [name.strip() for name in open(args.ssd_labels).readlines()]
    ssd_net = create_mobilenetv1_ssd(len(ssd_class_names), is_test=True)
    ssd_net.load(args.ssd)
    ssd_predictor = create_mobilenetv1_ssd_predictor(ssd_net, candidate_size=200)

    tracker = ObjectTracker(ssd_class_names)
    state_qualifier = StateQualifier(args.parking_box)

    print("Loading CNN")
    plate_detector = LicensePlateDetector(
        (args.plate_size[0], args.plate_size[1], args.plate_size[2], args.plate_size[3]))
    ocr_dataset = CharDataset(args.cnn_dataset_root, args.cnn_labels)
    cnn_net = CNN()
    cnn_net.load_state_dict(torch.load(args.cnn, map_location=lambda storage, loc: storage))
    char_seg = CharSeg(
        (args.char_properties[0], args.char_properties[1], args.char_properties[2], args.char_properties[3]),
        args.char_properties[4])
    lpr = LPR(char_seg, plate_detector, cnn_net, ocr_dataset)

    car_system = CarSystem(ssd_predictor, state_qualifier, tracker, lpr, args.frame_skip,
                           args.light_th, args.ssd_prob_th, args.license_plate)

    frame_num = 0
    while True:
        _, orig_image = cap.read()
        if orig_image is None:
            break
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        ids, boxes, labels, probabilities, state_dict, is_parked = car_system.handle_frame(image)
        print("Frame:", frame_num)
        print("Found", boxes.shape[0], "objects,", ids.shape[0], "of them are cars")

        for i in range(0, boxes.shape[0]):
            if i < ids.shape[0]:
                print('Car ID:', ids[i], "box:", boxes[i])
            else:
                print('Object box:', boxes[i])

        for id in ids:
            if state_dict[id] == 'ARRIVED':
                print("Found car with ID:", id, 'ARRIVED')
                if is_parked:
                    print("LICENSE_PLATE: IDENTIFIED")
            elif state_dict[id] == 'LEFT':
                print("Found car with ID:", id, 'LEFT')

        print("")
        frame_num = frame_num + 1
    cap.release()
