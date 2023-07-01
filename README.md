# BrainhackFinals
This repository documents the work that my team and I have done for Brainhack 2023: Today-I-Learned. We trained an Object Detection Model on YOLOv8, Re-Identification Model using Siamese Network and Speech Recognition Model.
We implemented our Artificial Intelligence models into a DJI Robomaster to enable it to perform AI tasks. On the robot, we implemented A* Search for path-planning and holonomic drive. 

## Object Detection
The code to train our Object Detection and Re-Identification Model can be found in the "Brainhack_TIL_ObjectDetection.ipynb" file. We used YOLOv8 to train the portion on object detection. For re-identification, we implemented
a Siamese Network. The Siamese Network used Triplet Loss and Cosine Similarity as its loss function, which performed better than Contrastive Loss and Euclidean Distance.

## Speech Recognition

## Robot Planning
The code for robot planning can be found under any of the arena folders. The stubs folder contains all the code necessary to run the robot. As an overview, the robot uses A* Search to plan its route, with the planning done using the planner.py file. To perform the AI tasks, it calls on the different services (ReID, Speech Recognition, Speaker Identification). 
