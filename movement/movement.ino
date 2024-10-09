// Pin definitions for motor control
const int motor1In1 = 9;
const int motor1In2 = 8;
const int motor2In1 = 7;
const int motor2In2 = 6;
const int ena = 10;  // Speed control for motor 1
const int enb = 5;   // Speed control for motor 2

// Joystick pin definitions
const int joystick1X = A0; // Joystick 1 X-axis (for turning)
const int joystick1Y = A1; // Joystick 1 Y-axis (for forward/backward)

const int joystick2X = A2; // Joystick 2 X-axis (for fine turning)
const int joystick2Y = A3; // Joystick 2 Y-axis (for fine speed control)

bool autonomousMode = true; // Start in autonomous mode

void setup() {
  pinMode(motor1In1, OUTPUT);
  pinMode(motor1In2, OUTPUT);
  pinMode(motor2In1, OUTPUT);
  pinMode(motor2In2, OUTPUT);
  pinMode(ena, OUTPUT);
  pinMode(enb, OUTPUT);

  Serial.begin(9600);
}

void loop() {
  // Check for joystick input to switch to manual control
  int joy1YValue = analogRead(joystick1Y);
  int joy1XValue = analogRead(joystick1X);
  int centerThreshold = 20;

  if (abs(joy1YValue) > centerThreshold || abs(joy1XValue) > centerThreshold) {
    autonomousMode = false; // Switch to manual control
  } else {
    autonomousMode = true; // Return to autonomous mode
  }

  if (autonomousMode) {
    // Autonomous movement: move forward
    int motor1Speed = 255; // Full speed for forward movement
    int motor2Speed = 255; // Full speed for forward movement

    // Control motor 1 (propeller 1) - Forward
    digitalWrite(motor1In1, HIGH);
    digitalWrite(motor1In2, LOW);
    analogWrite(ena, motor1Speed);   // Set speed

    // Control motor 2 (propeller 2) - Forward
    digitalWrite(motor2In1, HIGH);
    digitalWrite(motor2In2, LOW);
    analogWrite(enb, motor2Speed);   // Set speed

    // Read from serial to check for plastic detection command
    if (Serial.available()) {
      char command = Serial.read();
      if (command == '2') { // Move towards plastic
        // Here, we assume '2' means move forward towards detected plastic
        analogWrite(ena, motor1Speed);
        analogWrite(enb, motor2Speed);
      }
      else if (command == '3') { // Stop moving
        digitalWrite(motor1In1, LOW);
        digitalWrite(motor1In2, LOW);
        digitalWrite(motor2In1, LOW);
        digitalWrite(motor2In2, LOW);
      }
    }
  } else {
    // Manual control via joystick
    int motor1Speed = map(joy1YValue, 0, 1023, -255, 255);
    int motor2Speed = map(joy1YValue, 0, 1023, -255, 255);
    int turningControl = map(joy1XValue, 0, 1023, -255, 255);

    motor1Speed = motor1Speed + turningControl;
    motor2Speed = motor2Speed - turningControl;

    motor1Speed = constrain(motor1Speed, -255, 255);
    motor2Speed = constrain(motor2Speed, -255, 255);

    // Control motor 1 (propeller 1) - Forward/Backward
    if (abs(motor1Speed) > centerThreshold) {
      if (motor1Speed > 0) {
        digitalWrite(motor1In1, HIGH);
        digitalWrite(motor1In2, LOW);
      } else {
        digitalWrite(motor1In1, LOW);
        digitalWrite(motor1In2, HIGH);
        motor1Speed = -motor1Speed; // Make speed positive for PWM
      }
      analogWrite(ena, motor1Speed);
    } else {
      digitalWrite(motor1In1, LOW);
      digitalWrite(motor1In2, LOW);
    }

    // Control motor 2 (propeller 2) - Forward/Backward
    if (abs(motor2Speed) > centerThreshold) {
      if (motor2Speed > 0) {
        digitalWrite(motor2In1, HIGH);
        digitalWrite(motor2In2, LOW);
      } else {
        digitalWrite(motor2In1, LOW);
        digitalWrite(motor2In2, HIGH);
        motor2Speed = -motor2Speed; // Make speed positive for PWM
      }
      analogWrite(enb, motor2Speed);
    } else {
      digitalWrite(motor2In1, LOW);
      digitalWrite(motor2In2, LOW);
    }
  }
}
