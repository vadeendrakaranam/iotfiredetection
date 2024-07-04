import RPi.GPIO as GPIO
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import spidev
import cv2
import numpy as np
import tensorflow as tf

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
FLAME_SENSOR_PIN = 27
MQ7_PIN = 17
GPIO.setup(FLAME_SENSOR_PIN, GPIO.IN)
GPIO.setup(MQ7_PIN, GPIO.IN)

# Initialize SPI for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

# Load your TensorFlow Lite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load your pre-trained TensorFlow Lite model
model_path = '/path/to/your/model.tflite'  # Replace with your model path
interpreter = load_tflite_model(model_path)

# Function to read ADC value from MCP3008
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Function to capture image from webcam
def capture_image(image_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
    cap.release()
    return frame if ret else None

# Function to analyze flame using TensorFlow Lite model
def analyze_flame(interpreter, image_path, threshold=0.5):
    # Load and preprocess image
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))  # Resize to model input size
    image_normalized = image_resized / 255.0  # Normalize pixel values
    input_data = np.expand_dims(image_normalized, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    flame_probability = output_data[0][0]

    # Interpret output (assuming binary classification)
    flame_detected = flame_probability > threshold
    return flame_detected

# Function to send email with optional attachment
def send_email(image_path, alert_type):
    fromaddr = "karanam.vadeendra123456@gmail.com"
    toaddr = "karanam.radhamma123456@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Fire Alert!"

    body = alert_type
    msg.attach(MIMEText(body, 'plain'))

    if image_path:
        attachment = open(image_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= " + image_path)
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "Vadeendra@123")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

try:
    while True:
        if not GPIO.input(MQ7_PIN) and not GPIO.input(FLAME_SENSOR_PIN):
            print("No smoke or flame detected by sensors. Opening webcam to check...")
            image_path = "/home/pi/flame_detection.jpg"
            frame = capture_image(image_path)
            if frame is not None:
                flame_detected = analyze_flame(interpreter, image_path)
                if flame_detected:
                    alert_type = "No smoke or flame detected by sensors. Flame detected by webcam."
                    send_email(image_path, alert_type)
                    print("Alert sent!")
                else:
                    print("No flame detected by webcam.")
        elif GPIO.input(FLAME_SENSOR_PIN):
            print("Flame detected by flame sensor.")
            time.sleep(5)  # Sleep to avoid continuous alerts
        elif GPIO.input(MQ7_PIN):
            print("Smoke detected by MQ7 sensor.")
            time.sleep(5)  # Sleep to avoid continuous alerts
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()