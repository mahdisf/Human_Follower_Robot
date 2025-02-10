import cv2
import time
import tensorflow as tf  # Import TensorFlow
import numpy as np  # Import NumPy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform

if platform == 'linux' or platform == 'raspberry':
    import RPi.GPIO as GPIO  # Import RPi.GPIO for GPIO control
    from picamera import PiCamera
    from picamera.array import PiRGBArray

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

if platform == 'linux' or platform == 'raspberry':
    # GPIO pin configuration
    EN_PIN1 = 9
    MOTOR1_PIN1 = 8
    MOTOR1_PIN2 = 7
    MOTOR2_PIN1 = 5
    MOTOR2_PIN2 = 4
    EN_PIN2 = 3
    TRIG_PIN = 13
    ECHO_PIN = 12

    # Setup GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(EN_PIN1, GPIO.OUT)
    GPIO.setup(MOTOR1_PIN1, GPIO.OUT)
    GPIO.setup(MOTOR1_PIN2, GPIO.OUT)
    GPIO.setup(MOTOR2_PIN1, GPIO.OUT)
    GPIO.setup(MOTOR2_PIN2, GPIO.OUT)
    GPIO.setup(EN_PIN2, GPIO.OUT)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)

    def move_forward():
        GPIO.output(EN_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR1_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR1_PIN2, GPIO.LOW)
        GPIO.output(EN_PIN2, GPIO.HIGH)
        GPIO.output(MOTOR2_PIN1, GPIO.LOW)
        GPIO.output(MOTOR2_PIN2, GPIO.HIGH)

    def move_backward():
        GPIO.output(EN_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR1_PIN1, GPIO.LOW)
        GPIO.output(MOTOR1_PIN2, GPIO.HIGH)
        GPIO.output(EN_PIN2, GPIO.HIGH)
        GPIO.output(MOTOR2_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR2_PIN2, GPIO.LOW)

    def turn_left():
        GPIO.output(EN_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR1_PIN1, GPIO.LOW)
        GPIO.output(MOTOR1_PIN2, GPIO.HIGH)
        GPIO.output(EN_PIN2, GPIO.HIGH)
        GPIO.output(MOTOR2_PIN1, GPIO.LOW)
        GPIO.output(MOTOR2_PIN2, GPIO.HIGH)

    def turn_right():
        GPIO.output(EN_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR1_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR1_PIN2, GPIO.LOW)
        GPIO.output(EN_PIN2, GPIO.HIGH)
        GPIO.output(MOTOR2_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR2_PIN2, GPIO.LOW)

    def stop_motors():
        GPIO.output(EN_PIN1, GPIO.LOW)
        GPIO.output(MOTOR1_PIN1, GPIO.LOW)
        GPIO.output(MOTOR1_PIN2, GPIO.LOW)
        GPIO.output(EN_PIN2, GPIO.LOW)
        GPIO.output(MOTOR2_PIN1, GPIO.LOW)
        GPIO.output(MOTOR2_PIN2, GPIO.LOW)

    def check_distance():
        GPIO.output(TRIG_PIN, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, GPIO.LOW)
        while GPIO.input(ECHO_PIN) == 0:
            pass
        start_time = time.time()
        while GPIO.input(ECHO_PIN) == 1:
            pass
        end_time = time.time()
        duration = end_time - start_time
        distance = (duration * 34300) / 2
        return distance > 100

class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.img1 = Image()
        self.layout.add_widget(self.img1)

        if platform == 'linux' or platform == 'raspberry':
            self.camera = PiCamera()
            self.camera.resolution = (640, 480)
            self.raw_capture = PiRGBArray(self.camera, size=(640, 480))
            self.stream = self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True)
        else:
            self.cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
            if not self.cap.isOpened():
                print("Failed to open camera")

        # Load labels and model for TensorFlow Lite
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

        return self.layout

    def set_camera_resolution(self):
        if platform != 'linux' and platform != 'raspberry':
            # Set camera resolution to a wide but low resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # height

    def send_command(self, command):
        if platform == 'linux' or platform == 'raspberry':
            if command == 'F':
                if check_distance():
                    move_forward()
                else:
                    stop_motors()
            elif command == 'B':
                move_backward()
            elif command == 'L':
                turn_left()
            elif command == 'R':
                turn_right()
            elif command == 'S':
                stop_motors()

    def update(self, dt):
        if platform == 'linux' or platform == 'raspberry':
            frame = next(self.stream).array
            self.raw_capture.truncate(0)
        else:
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

        # Convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # Display image from the texture
        self.img1.texture = image_texture

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
        if platform == 'linux' or platform == 'raspberry':
            # No need to change camera for PiCamera
            pass
        else:
            self.current_camera_index = (self.current_camera_index + 1) % 3
            self.cap.release()
            self.cap = cv2.VideoCapture(self.current_camera_index)
            if not self.cap.isOpened():
                self.current_camera_index = 0
                self.cap = cv2.VideoCapture(self.current_camera_index)
            print(f"Switched to camera {self.current_camera_index}")

    def on_stop(self):
        if platform == 'linux' or platform == 'raspberry':
            # Cleanup GPIO on exit
            GPIO.cleanup()

if __name__ == '__main__':
    CameraApp().run()