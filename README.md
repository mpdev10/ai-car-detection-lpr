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

  For object detection, I used Single Shot MultiBox Detector (SSD) based on MobileNetV1.
  Character recognition - CNN.
  
  
  
  
