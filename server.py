import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Disable OneDNN optimizations for TensorFlow (optional based on your requirements)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MODEL_PATH = 'new_tflite_model.tflite'

# Initialize the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    while True:
        # Define the desired frame size
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Get the expected shape from the model's input details
        input_shape = input_details[0]['shape']  # Typically [1, height, width, channels]

        # Extract height, width, and channel info from the expected input shape
        model_height, model_width, expected_channels = input_shape[1], input_shape[2], input_details[0]['shape'][3]

        # Resize the frame to the model's expected size
        resized_frame = cv2.resize(frame, (model_width, model_height))

        # Check if the model expects grayscale (1 channel) or RGB (3 channels)
        if expected_channels == 1:
            # Convert the image to grayscale
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            # Add channel dimension (since it's grayscale, we need to add a dimension for the single channel)
            resized_frame = np.expand_dims(resized_frame, axis=-1)
        elif expected_channels == 3:
            # Convert BGR to RGB
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Expand dimensions to match the model's input (e.g., add batch dimension)
        input_data = np.expand_dims(resized_frame, axis=0)

        # Convert to the correct data type (e.g., INT8 or FLOAT32 based on model)
        if input_details[0]['dtype'] == np.int8:
            input_data = input_data.astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)

        # Normalize the input data if required (e.g., [0, 1] for FLOAT32)
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data / 255.0

        # Set the tensor data for inference
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Invoke the interpreter to perform inference
        interpreter.invoke()

        # Get the model's prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # If the model uses quantization, dequantize the output
        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            output_data = scale * (output_data - zero_point)

        # Apply softmax to get confidence scores if not already applied
        confidence_scores = tf.nn.softmax(output_data[0]).numpy()

        # Print confidence scores or do further processing
        print("Confidence Scores:", confidence_scores)

        plastic_confidence = confidence_scores[0]  # Assume plastic is first class
        non_plastic_confidence = confidence_scores[1]

        # Define class labels based on your model's training
        class_labels = ['Plastic', 'Non-plastic']

        # Display confidence scores on the frame
        for i, label in enumerate(class_labels):
            cv2.putText(frame, f"{label}: {confidence_scores[i]:.2f}",
                        (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Provide visual feedback based on detection
        if plastic_confidence > non_plastic_confidence:
            cv2.putText(frame, "Plastic Detected", (10, 30 + len(class_labels)*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Plastic Detected", (10, 30 + len(class_labels)*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Plastic Recognition', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting the program.")
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")
