# si-projekt

System designed for car detection, car's state classification, and license plate recogniton.

## Functionalities

1. Detection of:

    * Cars
    * People 
    
2. Car state classification:
  
    * MOVE
    * ARRIVED
    * LEFT
    
3. Light state classification:
    
    * ON
    * OFF

3. License plate recognition:
    
    * Plate detection
    * Character segmentation
    * Character recognition
    
## Architecture

Code is written in python 3.7, using mostly NumPy, PyTorch and OpenCV.

For object detection, I used Single Shot MultiBox Detector (SSD) based on MobileNetV1. Character recognition - CNN.
  
<details>
<summary>
<b>car_system.py</b>    
</summary>
<p>
  
### CarSystem
class **CarSystem**(_detection_net, detection_predictor, parking_place_bbox, state_qualifier, car_tracker, lpr, frame_skip, light_level_th, prob_th_)

The class wraps up all modules of the system together.

Parameters | Description
---------- | -----------
detection_net: **SSD** | Network class used for detection of cars and others
detection_predictor: **Predictor** | Predictor class which runs forward propagation in SSD and returns bounding boxes and probabilities
state_qualifier: **StateQualifier** | State qualifier class which determines the state of a car
lpr: **LPR** | License plate recognition class responsible for license plate detection, character segmentation and character recognition
frame_skip: **int** | Number of frames to skip during processing the video in handle_frame method
light_level_th: **float** | When average value of pixels of image in grayscale is higher than this threshold, then the state of lights is set to _on_, and frame is processed
prob_th: **float** | When probability of object's detection is higher than this threshold, then the object is returned in handle_frame method

Public methods:
<details>
<summary>
   <b>handle_frame</b>(<i>image</i>)
</summary>
</details>
 
</p>
</details>
  
  
  
