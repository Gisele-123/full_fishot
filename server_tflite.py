import cv2
import numpy as np
import tflite_runtime.interpreter as tflite  # Use tflite_runtime instead of tensorflow
import serial
import time
import os
import serial.tools.list_ports

# Disable oneDNN optimizations (if not needed)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Path to your TensorFlow Lite model
MODEL_PATH = 'new_tflite_model.tflite'

# Constants
COMMAND_COOLDOWN = 6  # Cooldown for sending serial commands (in seconds)
PLASTIC_DETECTED_SIGNAL = b'1'
MOVE_FORWARD_SIGNAL = b'2'  # New signal for continuous forward motion when plastic is detected
STOP_FORWARD_SIGNAL = b'3'  # Signal to stop forward motion

# Initialize TFLite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Serial communication setup
SERIAL_PORT = 'COM8'
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# Initialize variables for command timing
last_command_time = 0
plastic_last_detected_time = 0

try:
    while True:
        # Define expected width and height for input image
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Get the expected shape from the model's input details
        input_shape = input_details[0]['shape']  # Typically [1, height, width, channels]

        # Extract height, width, and channel info from the expected input shape
        height, width, expected_channels = input_shape[1], input_shape[2], input_shape[3]

        # Resize the frame to the model's expected size
        resized_frame = cv2.resize(frame, (width, height))

        # Check if the model expects grayscale (1 channel) or RGB (3 channels)
        if expected_channels == 1:
            # Convert the image to grayscale
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            # Add channel dimension for grayscale image
            resized_frame = np.expand_dims(resized_frame, axis=-1)

        # Expand dimensions to match the model's input (e.g., add batch dimension)
        input_data = np.expand_dims(resized_frame, axis=0)

        # Convert to the correct data type (e.g., INT8)
        input_data = input_data.astype(np.int8)

        # Set the tensor data for inference
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensor data (predictions)
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Convert int8 output to float32 for softmax
        output_data = output_data.astype(np.float32)

        # Apply softmax to get confidence scores
        confidence_scores = np.exp(output_data[0]) / np.sum(np.exp(output_data[0]))

        # Print confidence scores
        print("Confidence Scores:", confidence_scores)

        # Extract plastic and non-plastic confidence scores
        plastic_confidence = confidence_scores[0]  # Assume plastic is first class
        non_plastic_confidence = confidence_scores[1]

        # Display detection results on screen
        class_labels = ['Plastic', 'Non-plastic']
        for i, label in enumerate(class_labels):
            cv2.putText(frame, f"{label}: {confidence_scores[i]:.2f}",
                        (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show video frame with labels
        cv2.imshow('Plastic Recognition', frame)

        # Get current time for command cooldown
        current_time = time.time()

        # If plastic confidence is higher, send movement commands
        if plastic_confidence > non_plastic_confidence:
            # Plastic detected, send movement command if cooldown has passed
            if current_time - last_command_time > COMMAND_COOLDOWN:
                ser.write(PLASTIC_DETECTED_SIGNAL)
                last_command_time = current_time
                plastic_last_detected_time = current_time
                print("Plastic detected. Sent '1' to Arduino.")

            # Keep sending forward motion signal while plastic is detected
            ser.write(MOVE_FORWARD_SIGNAL)
            print("Continuing forward motion.")
        else:
            # Stop forward motion if no plastic is detected for 5 seconds
            if current_time - plastic_last_detected_time > 5:
                ser.write(STOP_FORWARD_SIGNAL)
                print("Plastic not detected, stopping forward motion.")

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("Serial connection closed.")
