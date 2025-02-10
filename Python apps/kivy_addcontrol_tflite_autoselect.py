import cv2
import time
import tensorflow as tf  # Import TensorFlow
import os  # Import os for path operations
import numpy as np  # Import NumPy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import serial
import serial.tools.list_ports
from kivy.utils import platform
from kivy.uix.label import Label
from kivy.core.audio import SoundLoader
from kivy.uix.button import Button

if platform == 'android':
    from android.permissions import request_permissions, Permission
    from jnius import autoclass

# TensorFlow Lite model paths
MODEL_PATH = "model/detect.tflite"
LABEL_PATH = "model/labelmap.txt"

def load_labels(label_path):
    """Loads the labels from the labelmap file."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = [line.strip() for line in lines]
    return labels

def load_model(model_path):
    """Loads the TFLite model and allocates tensors using TensorFlow."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def detect_objects(interpreter, image, input_details, output_details, labels, min_confidence_threshold=0.5):
    """Performs object detection on the input image and filters for 'person'."""

    # Resize the image to match input shape and convert to RGB
    input_shape = input_details[0]['shape']
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb_image, axis=0)
    input_data = input_data.astype(np.uint8)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    count = int(interpreter.get_tensor(output_details[3]['index'])[0])

    detections = []
    for i in range(count):
        if scores[i] > min_confidence_threshold:
            class_id = int(classes[i])
            label = labels[class_id] if class_id < len(labels) else f"Unknown ({class_id})"
            confidence = float(scores[i])
            if label == "person":  # Filter for "person" class
                ymin, xmin, ymax, xmax = boxes[i]
                im_height, im_width, _ = image.shape
                xmin = int(xmin * im_width)
                xmax = int(xmax * im_width)
                ymin = int(ymin * im_height)
                ymax = int(ymax * im_height)

                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })
    return detections

def draw_detections(image, detections, selected_person_id=None):
    """Draws bounding boxes and labels on the image.
       Only draws green boxes if no person is selected, otherwise only selected person is blue.
    """
    if selected_person_id is None: # Draw green boxes for all if no person selected
        for person_id, detection in enumerate(detections):
            xmin, ymin, xmax, ymax = detection[0] # Now detection is tuple (bbox, confidence)
            confidence = detection[1]
            label = f"Person {person_id + 1} ({confidence:.2f})"
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def request_android_permissions():
    # List of permissions your app needs
    permissions = [
        Permission.CAMERA,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.INTERNET,
    ]

    # Request each permission
    request_permissions(permissions)

class CameraApp(App):
    def build(self):
        # Request permissions on Android
        if platform == 'android':
            request_android_permissions()

        self.layout = BoxLayout(orientation='vertical')
        self.img1 = Image()
        self.layout.add_widget(self.img1)

        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
        if not self.cap.isOpened():
            print("Failed to open camera")

        # Load labels and model for TensorFlow Lite
        if not os.path.exists("model"):
            os.makedirs("model")
            print("Created 'model' directory. Please place 'detect.tflite' and 'labelmap.txt' inside the 'model' directory.")
            return BoxLayout(orientation='vertical', children=[Label(text="Model files not found. Please check console.")]) # Return basic layout to prevent crash

        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
            print(f"Error: Model file '{MODEL_PATH}' or label file '{LABEL_PATH}' not found. "
                  f"Make sure they are placed in the 'model' directory.")
            return BoxLayout(orientation='vertical', children=[Label(text="Model files not found. Please check console.")]) # Return basic layout to prevent crash


        self.labels = load_labels(LABEL_PATH)
        self.interpreter = load_model(MODEL_PATH)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.detected_persons = []
        self.selected_person_id = None
        self.person_selected = False
        self.fps_list = []

        # Set camera resolution based on device capabilities
        self.set_camera_resolution()

        # Schedule the update method to run at 60 frames per second
        Clock.schedule_interval(self.update, 1.0/60.0)

        # Enable touch handling for person selection
        self.img1.bind(on_touch_down=self.on_touch_down)

        self.last_position_command = 'S'
        self.last_movement_command = 'S'

        # Add label for Arduino connection status
        self.status_label = Label(text="Connecting to Arduino...", size_hint=(1, 0.1))
        self.layout.add_widget(self.status_label)

        # Load the sounds for connection and disconnection alerts
        self.connect_sound = SoundLoader.load('connect_alert.mp3')
        self.disconnect_sound = SoundLoader.load('connect_alert.mp3')

        # Attempt to connect to Arduino
        self.ser = None
        self.connect_to_arduino()

        # Schedule the connection check to run every second
        Clock.schedule_interval(self.check_connection, 0.1)

        # Add button to change camera
        self.change_camera_button = Button(text="Change Camera", size_hint=(1, 0.1))
        self.change_camera_button.bind(on_press=self.change_camera)
        self.layout.add_widget(self.change_camera_button)

        self.current_camera_index = 0

        return self.layout

    def set_camera_resolution(self):
        # Set camera resolution to a wide but low resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # height


    def connect_to_arduino(self):
        if platform == 'android':
            # Set the window size for your Samsung A34 phone (1080x2340)
            Window.size = (2340,1080)
            from usb4a import usb
            from usbserial4a import serial4a

            usb_device_list = usb.get_usb_device_list()
            if not usb_device_list:
                self.status_label.text = "No USB device found"
                print("No USB device found")
                return

            for device in usb_device_list:
                try:
                    self.ser = serial4a.get_serial_port(device.getDeviceName(), 9600, 8, 'N', 1, timeout=1)
                    if self.ser:
                        self.status_label.text = f"Connected to Arduino on {device.getDeviceName()}"
                        print(f"Connected to Arduino on {device.getDeviceName()}")
                        if self.connect_sound:
                            self.connect_sound.play()
                        return
                except Exception as e:
                    print(f"Failed to connect on {device.getDeviceName()}: {e}")
            self.status_label.text = "Failed to connect to any Arduino device"
            print("Failed to connect to any Arduino device")
        else:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                try:
                    self.ser = serial.Serial(port.device, 9600, timeout=1)
                    self.status_label.text = f"Connected to Arduino on {port.device}"
                    print(f"Connected to Arduino on {port.device}")
                    if self.connect_sound:
                        self.connect_sound.play()
                    return
                except serial.SerialException as e:
                    print(f"Failed to connect on {port.device}: {e}")
            self.status_label.text = "Failed to connect to any Arduino device"
            print("Failed to connect to any Arduino device")

    def send_command(self, command):
        if self.ser:
            time.sleep(0.1)
            try:
                self.ser.write(command.encode())
                print(f"Sent command: {command}")
            except serial.SerialException as e:
                print(f"Error sending command: {e}")

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        start_time = time.time()

        # Detect persons in the frame using TensorFlow Lite model
        detections = detect_objects(self.interpreter, frame, self.input_details, self.output_details, self.labels)
        self.detected_persons = [] # Clear previous detections and repopulate with formatted data
        for detection in detections:
            bbox = (detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax'])
            confidence = detection['confidence']
            self.detected_persons.append((bbox, confidence))

        # Automatically select the first detected person if no person is currently selected
        if not self.person_selected and self.detected_persons:
            self.selected_person_id = 1
            self.person_selected = True

        position_text = "center"
        movement_text = "stop"
        if self.person_selected and self.selected_person_id is not None:
            # Check if the selected person is still in the frame
            if self.selected_person_id <= len(self.detected_persons):
                selected_person_bbox = self.detected_persons[self.selected_person_id - 1][0] # Get bbox of selected person
                x1, y1, x2, y2 = selected_person_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box for selected person
                cv2.putText(frame, f'Selected Person {self.selected_person_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Determine the middle rectangle
                frame_height, frame_width, _ = frame.shape
                mid_x1 = frame_width // 3
                mid_x2 = 2 * frame_width // 3

                # Check if the selected person is to the left or right of the middle rectangle
                person_center_x = (x1 + x2) // 2
                if person_center_x < mid_x1:
                    position_text = "left"
                    if self.last_position_command != 'L':
                        self.send_command('L')
                        self.last_position_command = 'L'
                elif person_center_x > mid_x2:
                    position_text = "right"
                    if self.last_position_command != 'R':
                        self.send_command('R')
                        self.last_position_command = 'R'
                else:
                    position_text = "center"
                    if self.last_position_command != 'S':
                        self.send_command('S')
                        self.last_position_command = 'S'

                # Check if the selected person is in the top rectangle
                if y1 < frame_height // 4:
                    movement_text = "backward"
                    if self.last_movement_command != 'B':
                        self.send_command('B')
                        self.last_movement_command = 'B'
                elif y1 < frame_height // 2:
                    movement_text = "stop"
                    if self.last_movement_command != 'S':
                        self.send_command('S')
                        self.last_movement_command = 'S'
                else:
                    movement_text = "forward"
                    if self.last_movement_command != 'F':
                        self.send_command('F')
                        self.last_movement_command = 'F'
            else:
                # If the selected person is not detected, turn around in the last known direction
                if self.last_position_command == 'L':
                    self.send_command('L')  # Turn left
                elif self.last_position_command == 'R':
                    self.send_command('R')  # Turn right
        else:
            # Default to stop and center if no person is selected
            if self.last_position_command != 'S':
                self.send_command('S')
                self.last_position_command = 'S'
            if self.last_movement_command != 'S':
                self.send_command('S')
                self.last_movement_command = 'S'

        # Draw bounding boxes (green for all persons if none selected, blue for selected)
        frame = draw_detections(frame, self.detected_persons, self.selected_person_id if self.person_selected else None)


        # Draw 9x9 grid
        frame_height, frame_width, _ = frame.shape
        num_lines = 5
        for i in range(1, num_lines):
            # Vertical lines
            x = i * frame_width // num_lines
            cv2.line(frame, (x, 0), (x, frame_height), (255, 255, 255), 1)
            # Horizontal lines
            y = i * frame_height // num_lines
            cv2.line(frame, (0, y), (frame_width, y), (255, 255, 255), 1)

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        self.fps_list.append(fps)
        if len(self.fps_list) > 30:
            self.fps_list.pop(0)
        avg_fps = sum(self.fps_list) / len(self.fps_list)
        # Display FPS, position, and movement on frame
        text_color = (0, 0, 255)  # Red color for better visibility
        cv2.putText(frame, f'FPS: {avg_fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, position_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, movement_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # Display Arduino connection status
        self.check_connection(0)

        # Convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # Display image from the texture
        self.img1.texture = image_texture

    def check_connection(self, dt):
        previous_status = self.status_label.text

        # Check if the current connection is still valid
        if self.ser:
            try:
                # Check if the serial port is still open
                self.ser.in_waiting
                self.status_label.text = "Arduino is connected."
                self.status_label.color = (0, 1, 0, 1)  # Green background
                if previous_status != "Arduino is connected.":
                    if self.connect_sound:
                        self.connect_sound.play()
                    print("Arduino is connected.")
            except (OSError, serial.SerialException):
                # Handle disconnection
                self.status_label.text = "Arduino is not connected."
                self.status_label.color = (1, 0, 0, 1)  # Red background
                if previous_status != "Arduino is not connected.":
                    if self.disconnect_sound:
                        self.disconnect_sound.play()
                    print("Arduino is not connected.")
                # Close the serial connection to ensure a clean state
                if self.ser:
                    self.ser.close()
                self.ser = None  # Reset the serial object

        else:
            # If no connection exists, attempt to reconnect
            self.status_label.text = "Attempting to reconnect..."
            self.status_label.color = (1, 1, 0, 1)  # Yellow background
            self.connect_to_arduino()  # Try to reconnect


    def on_touch_down(self, instance, touch):
        # This method will handle the touch event for selecting a person
        if self.img1.collide_point(*touch.pos):
            # Get the size and position of the image widget
            img_x, img_y = self.img1.texture.size
            norm_x = touch.x / img_x
            norm_y = 1 - touch.y / img_y  # In Kivy, (0,0) is at the bottom left

            # Map normalized coordinates to frame coordinates
            frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            click_x = int(norm_x * frame_width)
            click_y = int(norm_y * frame_height)

            # Select the person based on the click position
            for person_id, (bbox, _) in enumerate(self.detected_persons):
                x1, y1, x2, y2 = bbox
                # Corrected condition: use click_y for y-coordinate check
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    self.selected_person_id = person_id + 1  # Start counting from 1
                    self.person_selected = True
                    break
            else: # if no person is selected with the touch, unselect person.
                self.person_selected = False
                self.selected_person_id = None

    def change_camera(self, instance):
        self.current_camera_index = (self.current_camera_index + 1) % 3
        self.cap.release()
        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            self.current_camera_index = 0
            self.cap = cv2.VideoCapture(self.current_camera_index)
        print(f"Switched to camera {self.current_camera_index}")

if __name__ == '__main__':
    CameraApp().run()