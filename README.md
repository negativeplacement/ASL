For best results when testing use the demo.xml it was built using the signAlphaSet Dataset which is greatly larger than the dataset used to build the mediapipe_model.xml

This application uses OpenCV DNN and MediaPipe landmark models to convert American Sign Language (ASL) into text in real-time. It features a custom SVM (Support Vector Machine) classifier trained on hand landmark data.

Tools:
C++17 Compiler
CMake 3.15 or higher
OpenCV
Webcam

Required Files
Ensure the following models are in the same directory as your executable:
palm_detection_mediapipe_2023feb.onnx
handpose_estimation_mediapipe_2023feb.onnx
demo_model.xml (The trained SVM model)

Build Instructions

Extract the project.
Open a terminal in the project root.
Run the build commands:
bash
mkdir build
cd build
cmake ..
cmake --build .
<video src="Demo1.mp4" width="100%"></video>
