import re

cpp_file = 'person_detect_model_data.cpp'
tflite_file = 'model.tflite'

with open(cpp_file, 'r') as f:
    data = f.read()

# Extract the byte array
bytes_str = re.findall(r'unsigned char .*? = \{([^}]+)\};', data, re.S)
if bytes_str:
    bytes_str = bytes_str[0].replace('\n', '').replace(' ', '')
    byte_values = bytes_str.split(',')

    with open(tflite_file, 'wb') as out_file:
        for byte in byte_values:
            if byte:  # Avoid empty strings
                out_file.write(bytes([int(byte, 16)]))
    print(f"Extracted TFLite model to {tflite_file}")
else:
    print("Failed to find byte array in the C++ file.")
