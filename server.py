import cv2
import numpy as np
import tensorflow as tf
import serial
import time
import os
import serial.tools.list_ports

# Disable OneDNN optimizations for TensorFlow (optional based on your requirements)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MODEL_PATH = 'new_tflite_model.tflite'

COMMAND_COOLDOWN = 6  # Cooldown for sending serial commands (in seconds)
PLASTIC_DETECTED_SIGNAL = b'1'
MOVE_FORWARD_SIGNAL = b'2'  # New signal for continuous forward motion when plastic is detected
STOP_FORWARD_SIGNAL = b'3'  # Signal to stop forward motion

# Initialize the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to list available serial ports and let the user select one
def select_serial_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found. Please connect your Arduino and try again.")
        exit()

    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"{i + 1}: {port.device} - {port.description}")

    while True:
        try:
            selection = input("Select the port number where Arduino is connected (e.g., 1): ")
            selected_index = int(selection) - 1
            if selected_index < 0 or selected_index >= len(ports):
                raise ValueError
            selected_port = ports[selected_index].device
            print(f"Selected port: {selected_port}")
            return selected_port
        except ValueError:
            print("Invalid selection. Please enter a valid port number.")

# Let the user select the serial port
SERIAL_PORT = select_serial_port()
BAUD_RATE = 9600

# Initialize serial communication with the selected port
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for the serial connection to initialize
    print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Failed to connect to {SERIAL_PORT}: {e}")
    exit()

# Initialize the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    ser.close()
    exit()

last_command_time = 0  # Last time a command was sent
plastic_last_detected_time = 0  # Keep track of last plastic detection

try:
    while True:
        # Define the desired frame size
        width, height = 224, 224  # Adjusted to a square shape for many models
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Get the expected shape from the model's input details
        input_shape = input_details[0]['shape']  # Typically [1, height, width, channels]

        # Extract height, width, and channel info from the expected input shape
        model_height, model_width, expected_channels = input_shape[1], input_shape[2], input_shape[3]

        # Resize the frame to the model's expected size
        resized_frame = cv2.resize(frame, (model_width, model_height))

        # Check if the model expects grayscale (1 channel) or RGB (3 channels)
        if expected_channels == 1:
            # Convert the image to grayscale
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            # Add channel dimension (since it's grayscale, we need to add a dimension for the single channel)
            resized_frame = np.expand_dims(resized_frame, axis=-1)

        # Expand dimensions to match the model's input (e.g., add batch dimension)
        input_data = np.expand_dims(resized_frame, axis=0)

        # Convert to the correct data type (e.g., INT8)
        input_data = input_data.astype(np.int8)

        # Set the tensor data for inference
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Invoke the interpreter to perform inference
        interpreter.invoke()

        # Assuming output_data is the model's prediction (currently int8)
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Convert int8 output to float32
        output_data = output_data.astype(np.float32)

        # Apply softmax to get confidence scores
        confidence_scores = tf.nn.softmax(output_data[0]).numpy()

        # Print confidence scores or do further processing
        print("Confidence Scores:", confidence_scores)

        plastic_confidence = confidence_scores[0]  # Assume plastic is first class
        non_plastic_confidence = confidence_scores[1]

        # Show detection on screen
        class_labels = ['Plastic', 'Non-plastic']
        for i, label in enumerate(class_labels):
            cv2.putText(frame, f"{label}: {confidence_scores[i]:.2f}",
                        (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Plastic Recognition', frame)

        current_time = time.time()

        if plastic_confidence > non_plastic_confidence:
            # Plastic detected, send movement command
            if current_time - last_command_time > COMMAND_COOLDOWN:
                ser.write(PLASTIC_DETECTED_SIGNAL)
                last_command_time = current_time
                plastic_last_detected_time = current_time
                print("Plastic detected. Sent '1' to Arduino.")

            # Keep moving forward autonomously if plastic is still detected
            ser.write(MOVE_FORWARD_SIGNAL)
            print("Continuing forward motion.")
        else:
            # If plastic not detected for a certain time, stop forward motion
            if current_time - plastic_last_detected_time > 5:  # 5-second buffer
                ser.write(STOP_FORWARD_SIGNAL)
                print("Plastic not detected, stopping forward motion.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting the program.")
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("Serial connection closed.")
