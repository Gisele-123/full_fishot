#include <Servo.h>

const int servoPin = 4;
const int motor1In1 = 9;
const int motor1In2 = 8;
const int motor2In1 = 7;
const int motor2In2 = 6;
const int ena = 10;
const int enb = 5;

Servo myServo;

const int OPEN_POSITION = 120;
const int CLOSED_POSITION = 0;
const unsigned long HOLD_DURATION = 2000;

bool isAutonomous = false;  // Keep track of AUV/ROV mode

void setup() {
  Serial.begin(9600);
  myServo.attach(servoPin);
  myServo.write(CLOSED_POSITION);

  pinMode(motor1In1, OUTPUT);
  pinMode(motor1In2, OUTPUT);
  pinMode(motor2In1, OUTPUT);
  pinMode(motor2In2, OUTPUT);
  pinMode(ena, OUTPUT);
  pinMode(enb, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    // Plastic detection command from Python script
    if (command == '1') {
      Serial.println("Plastic detected! Opening servo.");
      myServo.write(OPEN_POSITION);
      delay(HOLD_DURATION);
      myServo.write(CLOSED_POSITION);
    }
    
    // Autonomous forward motion command
    else if (command == '2') {
      Serial.println("Autonomous forward motion.");
      moveForward();
      isAutonomous = true;  // AUV mode
    }
    
    // Stop forward motion command
    else if (command == '3') {
      Serial.println("Stopping forward motion.");
      stopMotors();
      isAutonomous = false;
    }
  }

  // Joystick controls for ROV mode
  if (!isAutonomous) {
    // Read joystick 1 for forward/backward and turning
    int joy1YValue = analogRead(A1);
    int joy1XValue = analogRead(A0);

    // Map values and control motors
    int motor1Speed = map(joy1YValue, 0, 1023, -255, 255);
    int motor2Speed = motor1Speed;
    int turningControl = map(joy1XValue, 0, 1023, -255, 255);

    motor1Speed += turningControl;
    motor2Speed -= turningControl;
    
    controlMotors(motor1Speed, motor2Speed);
  }
}

void moveForward() {
  // Autonomous forward movement logic
  digitalWrite(motor1In1, HIGH);
  digitalWrite(motor1In2, LOW);
  analogWrite(ena, 255);
  digitalWrite(motor2In1, HIGH);
  digitalWrite(motor2In2, LOW);
  analogWrite(enb, 255);
}

void stopMotors() {
  // Stop the motors
  digitalWrite(motor1In1, LOW);
  digitalWrite(motor1In2, LOW);
  digitalWrite(motor2In1, LOW);
  digitalWrite(motor2In2, LOW);
}

void controlMotors(int motor1Speed, int motor2Speed) {
  if (motor1Speed > 0) {
    digitalWrite(motor1In1, HIGH);
    digitalWrite(motor1In2, LOW);
  } else {
    digitalWrite(motor1In1, LOW);
    digitalWrite(motor1In2, HIGH);
    motor1Speed = -motor1Speed;
  }
  analogWrite(ena, motor1Speed);

  if (motor2Speed > 0) {
    digitalWrite(motor2In1, HIGH);
    digitalWrite(motor2In2, LOW);
  } else {
    digitalWrite(motor2In1, LOW);
    digitalWrite(motor2In2, HIGH);
    motor2Speed = -motor2Speed;
  }
  analogWrite(enb, motor2Speed);
}
