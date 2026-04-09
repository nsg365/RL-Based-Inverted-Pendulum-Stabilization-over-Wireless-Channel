const long BAUD = 115200;
const unsigned long sampleInterval = 10; 

// u = -Kx
float k1 = -4.4681;
float k2 = -1.6210; 
float k3 = -39.5037; 
float k4 = -5.0402; 
         

unsigned long lastTime = 0;

// Pins
const int armPinA = 20; const int armPinB = 21;
const int pendPinA = 19; const int pendPinB = 18;

// Motor Driver Pins (Arm Motor)
const int motArmPWM = 5; // PWM Pin
const int motArmDir = 4; // Direction Pin

//Power
const float SUPPLY = 12.0;

// Volatile counts for ISR
volatile long armCount = 0;
volatile long pendCount = 0;

//Filtering
const float alpha = 0.8;

// State variables
long armlastCount = 0;
long pendlastCount = 0;
float armVel = 0;
float pendVel = 0;

float u = 0;
float u_real = 0;

const int totalTicks = 2000;
const int halfTicks = 1000; 
const float TICK_TO_RAD = 3.1415926535 / 1000.0;

float filarmVel = 0;
float filpendVel = 0;

void setup() {
  Serial.begin(BAUD); // High baud rate is essential for low delay
  
  pinMode(armPinA, INPUT_PULLUP); pinMode(armPinB, INPUT_PULLUP);
  pinMode(pendPinA, INPUT_PULLUP); pinMode(pendPinB, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(armPinA), readEncoder1A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(armPinB), readEncoder1B, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pendPinA), readEncoder2A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pendPinB), readEncoder2B, CHANGE);

  // Setup Motor
  pinMode(motArmPWM, OUTPUT); 
  pinMode(motArmDir, OUTPUT);
  stopMotor();

  //k1 = 0.75 * k1; k2 = 0.25 * k2; k3 = 1.6 * k3; k4 = 0.25 * k4;  //First iteration

  //k1 = 0.75 * k1; k2 = 0.5 * k2; k3 = 1.55 * k3; k4 = 0.25 * k4;  //Second iteration

  k1 = 0.75 * k1; k2 = 0.5 * k2; k3 = 1.55 * k3; k4 = 0.4 * k4;  //Third iteration :)

  //k1 = 0.75 * k1; k2 = 0.5 * k2; k3 = 1.1 * k3; k4 = 0.4 * k4;  //loop time 15 ms :)

  //k1 = 0.75 * k1; k2 = 0.5 * k2; k3 = 0.8 * k3; k4 = 0.5 * k4;  //loop time 20 ms :)

  //k1 = 0.75 * k1; k2 = 0.5 * k2; k3 = 0.85 * k3; k4 = 0.5 * k4;  //loop time 20 ms :)

  //k1 = 0.75 * k1; k2 = 0.5 * k2; k3 = 0.75 * k3; k4 = 0.4 * k4;  //loop time 30 ms :)
}

void loop() {

  unsigned long currentTime = millis();

  if (currentTime - lastTime >= sampleInterval) {

    float dt = (currentTime - lastTime) / 1000.0;
    long armCurrent, pendCurrent;

    //Atomic Snapshots
    noInterrupts();
    armCurrent = armCount;
    pendCurrent = pendCount;
    interrupts();

    float darm = armCurrent - armlastCount;
    float dpend = pendCurrent - pendlastCount;

    if (darm > halfTicks) darm -= totalTicks;
    else if (darm <= -halfTicks) darm += totalTicks;
    
    if (dpend > halfTicks) dpend -= totalTicks;
    else if (dpend <= -halfTicks) dpend += totalTicks;

    float armVel = (darm * TICK_TO_RAD) / dt;
    float pendVel = (dpend * TICK_TO_RAD) / dt;

    filarmVel = (alpha * armVel) + ((1 - alpha) * filarmVel);
    filpendVel = (alpha * pendVel) + ((1 - alpha) * filpendVel);

    float arm = armCurrent * TICK_TO_RAD;
    float pend = pendCurrent * TICK_TO_RAD;

    if (abs(pend) < 0.35) { // ~20 degrees capture range
      u = -(k1 * arm + k2 * filarmVel + k3 * pend + k4 * filpendVel);
      //u_real = applyVoltage(u);
      applyVoltage(u);
    } else {
      // If the pendulum falls outside the capture range, turn off the motor for safety
      stopMotor(); 
    }

    Serial.print(arm, 3); Serial.print(","); 
    Serial.print(filarmVel, 2); Serial.print(",");
    Serial.print(pend, 3); Serial.print(",");
    Serial.print(filpendVel, 2); Serial.print(",");
    Serial.println(u); 

/*  Serial.print(u, 2); 
    Serial.print(",");
    Serial.println(u_real, 2);*/

    armlastCount = armCurrent;
    pendlastCount = pendCurrent;
    lastTime = currentTime;

  }

}

void readEncoder1A() {

  (digitalRead(armPinA) != digitalRead(armPinB)) ? armCount++ : armCount--;

  if  (armCount > 1000)
    armCount -= 2000;
  else if (armCount < -1000) 
    armCount += 2000;

}

void readEncoder1B() {

  (digitalRead(armPinA) == digitalRead(armPinB)) ? armCount++ : armCount--;

  if  (armCount > 1000)
    armCount -= 2000;
  else if (armCount < -1000) 
    armCount += 2000;

}

void readEncoder2A() {

  (digitalRead(pendPinA) != digitalRead(pendPinB)) ? pendCount++ : pendCount--;

  if  (pendCount > 1000)
    pendCount -= 2000;
  else if (pendCount < -1000) 
    pendCount += 2000;

}

void readEncoder2B() {

  (digitalRead(pendPinA) == digitalRead(pendPinB)) ? pendCount++ : pendCount--;

  if  (pendCount > 1000)
    pendCount -= 2000;
  else if (pendCount < -1000) 
    pendCount += 2000;

}

// --- MOTOR LOGIC ---
void applyVoltage(float volts) {
  // 1. Calculate Duty Cycle (-1.0 to 1.0)

  if (volts > 0.05) { 
    volts = volts + 1.41;
  } else if (volts < -0.05) {
    volts = volts - 1.41;
  } else {
    volts = 0; // Ignore sensor noise near zero to prevent motor buzzing
  }

  float duty = volts / SUPPLY;

  // 2. Clamp to safe limits
  if (duty > 1.0) duty = 1.0;
  if (duty < -1.0) duty = -1.0;
  
  // 3. Convert to PWM (0-255)
  int pwm = abs(duty) * 255;
  
  // 4. Direction Logic
  // If positive voltage moves arm WRONG way, swap HIGH/LOW here
  digitalWrite(motArmDir, (duty > 0) ? HIGH : LOW);
  analogWrite(motArmPWM, pwm);

  //return duty * SUPPLY;
}

void stopMotor() {
  analogWrite(motArmPWM, 0);
}