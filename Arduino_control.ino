// out 1 : green
// out4 : purpel
// Define motor control pins
const int enPin1 = 9; // ENA on the L298N
const int motor1Pin1 = 8; // IN1 on the L298N
const int motor1Pin2 = 7; // IN2 on the L298N
const int motor2Pin1 = 5; // IN3 on the L298N
const int motor2Pin2 = 4; // IN4 on the L298N
const int enPin2 = 3; // ENB on the L298N

// Defines Arduino pins for ultrasonic sensor
const int trigPin = 13;
const int echoPin = 12;
const int ultraon = 11; // Potentially unused pin
// Defines variables
long duration;
int distance;
int dis;

void setup() {

  // Set all the motor control pins to outputs
  pinMode(enPin1, OUTPUT);
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enPin2, OUTPUT);

  pinMode(ultraon, OUTPUT); // Potentially unused pin
  digitalWrite(ultraon, HIGH); // Potentially unused pin

  // Initialize ultrasonic sensor pins
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT);  // Sets the echoPin as an Input

  // Initialize serial communication
  Serial.begin(9600);
  Serial.println("System Ready");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    Serial.print("Received command: ");
    Serial.println(command);

    switch (command) {
      case 'F': // Move forward
        if (checkDistance()) {
          Serial.println("Moving Forward");
          moveForward();
        } else {
          Serial.println("Obstacle detected! Stopping.");
          stopMotors();
        }
        break;

      case 'B': // Move backward
        Serial.println("Moving Backward");
        moveBackward();
        break;

      case 'L': // Turn left
        Serial.println("Turning Left");
        turnLeft();
        break;

      case 'R': // Turn right
        Serial.println("Turning Right");
        turnRight();
        break;

      case 'S': // Stop
        Serial.println("Stopping");
        stopMotors();
        break;

      // --- ADDED CASES FOR '1' and '4' ---
      case '1':
        Serial.println("green"); // Output "green" for command '1'
        break;
      case '4':
        Serial.println("purple"); // Output "purple" for command '4' (corrected spelling)
        break;
      // --- END ADDED CASES ---

      default:
        Serial.println("Unknown command");
        break;
    }
  }
}

// Function to measure distance using ultrasonic sensor
bool checkDistance() {
  // Clears the trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);

  // Calculating the distance
  distance = duration * 0.034 / 2;

  // Print the distance for debugging purposes
  Serial.print("Distance from the object = ");
  Serial.print(distance);
  Serial.println(" cm");

  // Check if distance is greater than 100 cm
  return distance > 100;
}

// Motor movement functions
// Motor movement functions
void moveForward() {
  digitalWrite(enPin1, HIGH);
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(enPin2, HIGH);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
}

void moveBackward() {
  digitalWrite(enPin1, HIGH);
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(enPin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
}

void turnLeft() {
  digitalWrite(enPin1, HIGH);
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(enPin2, HIGH);
  digitalWrite(motor2Pin1, LOW); // Corrected motor2Pin1 to LOW
  digitalWrite(motor2Pin2, HIGH); // Corrected motor2Pin2 to HIGH
}

void turnRight() {
  digitalWrite(enPin1, HIGH);
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(enPin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
}

void stopMotors() {
  digitalWrite(enPin1, LOW);
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(enPin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
}
