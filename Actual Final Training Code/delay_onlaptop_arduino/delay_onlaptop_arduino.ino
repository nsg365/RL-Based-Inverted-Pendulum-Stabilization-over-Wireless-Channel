/*
 * RL_v2.ino — Hardened firmware for Furuta pendulum TD3/SAC deployment
 *
 * Fixes over RL.ino:
 * 1. Hardware Timer1 interrupt at exactly 100 Hz (no millis() jitter)
 * 2. Pendulum angle safety cutoff (stops motor if |theta| > SAFE_ANGLE)
 * 3. Removed ISR-level encoder wrap (let counts accumulate freely)
 * 4. Fixed dt to constant LOOP_MS/1000 instead of measured millis() delta
 * 5. Non-blocking serial receive with explicit stale-packet discard
 * 6. Binary packet format preserved (16 bytes out, 4 bytes in)
 * 7. UPDATED IIR Filter (alpha = 0.25) to match Python Augmented LQR
 * 8. CORRECTED Deadzone: 0.5V threshold and deadband to prevent overcompensation
 *
 * Target: Arduino Mega 2560 (Timer1, 16 MHz)
 */

#include <Arduino.h>

// ── Pin Configuration ────────────────────────────────────────────────────────
const int encArmPinA  = 20;
const int encArmPinB  = 21;
const int encPendPinA = 19;
const int encPendPinB = 18;
const int motArmPWM   = 5;
const int motArmDir   = 4;

// ── Sign Configuration (run diagnose.py to determine) ───────────────────────
const float PEND_SIGN  =  -1.0;
const float ARM_SIGN   =  1.0;
const float MOTOR_SIGN =  1.0;

// ── Physical Constants ──────────────────────────────────────────────────────
const float SUPPLY_VOLTAGE = 12.0;
const int   ENC_CPR        = 2000;
const float TICK_TO_RAD    = (2.0 * PI) / ENC_CPR;

// ── Timing ──────────────────────────────────────────────────────────────────
const int   LOOP_HZ        = 100;
const float DT             = 1.0 / LOOP_HZ;            // 0.01 s — fixed
const int   SAFETY_TIMEOUT = 100;                       // ms without command → stop

// ── Safety ──────────────────────────────────────────────────────────────────
// ~ degrees — must match FurutaEnv termination (0.4 rad)
const float SAFE_PEND_ANGLE = 0.4;

// ── Deadzone Compensation ───────────────────────────────────────────────────
const float DEADZONE = 0.0;
const float THRESH = 0.0; // Ignore sensor noise near zero

// ── Binary Packet (sent to Python every tick) ───────────────────────────────
struct StatePacket {
  float thetaPend;
  float thetaArm;
  float omegaPend;
  float omegaArm;
};

// ── Globals ─────────────────────────────────────────────────────────────────
volatile long countArm  = 0;   // free-running (no wrap in ISR)
volatile long countPend = 0;

long  lastCountArm       = 0;
long  lastCountPend      = 0;
float omegaArmFiltered   = 0.0;
float omegaPendFiltered  = 0.0;

volatile bool tickFlag       = false;   // set by Timer1 ISR
unsigned long lastCommandTime = 0;
bool safetyCutoff             = false;  // latched until reset

// ── Timer1 Setup (CTC mode, exact 100 Hz on 16 MHz Mega) ───────────────────
void setupTimer1() {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;

  // CTC mode: TOP = OCR1A
  // Prescaler = 256 → tick = 16 us
  // OCR1A = (16 000 000 / (256 * 100)) - 1 = 624
  OCR1A  = 624;
  TCCR1B = (1 << WGM12) | (1 << CS12);   // CTC + prescaler 256
  TIMSK1 = (1 << OCIE1A);                 // enable compare-match interrupt
  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  tickFlag = true;
}

// ── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(500000);

  // Encoders
  pinMode(encArmPinA,  INPUT_PULLUP);
  pinMode(encArmPinB,  INPUT_PULLUP);
  pinMode(encPendPinA, INPUT_PULLUP);
  pinMode(encPendPinB, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(encArmPinA),  isrArmA,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(encArmPinB),  isrArmB,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(encPendPinA), isrPendA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encPendPinB), isrPendB, CHANGE);

  // Motor
  pinMode(motArmPWM, OUTPUT);
  pinMode(motArmDir, OUTPUT);
  stopMotor();

  // Start deterministic 100 Hz tick
  setupTimer1();

  lastCommandTime = millis();
}

// ── Main Loop ───────────────────────────────────────────────────────────────
void loop() {

  // ── 1. Timer-driven sensor read + transmit (exactly 100 Hz) ──────────────
  if (tickFlag) {
    tickFlag = false;

    // Atomically snapshot encoder counts
    noInterrupts();
    long cArm  = countArm;
    long cPend = countPend;
    interrupts();

    // Tick deltas (no wrap handling needed — counters are free-running)
    long dArm  = cArm  - lastCountArm;
    long dPend = cPend - lastCountPend;

    // Velocity from discrete differentiation (fixed dt, no jitter)
    float rawVelArm  = (dArm  * TICK_TO_RAD) / DT;
    float rawVelPend = (dPend * TICK_TO_RAD) / DT;

    // First-order IIR low-pass (alpha = 0.25 to match Python script updates)
    omegaArmFiltered  = 0.25 * rawVelArm  + 0.75 * omegaArmFiltered;
    omegaPendFiltered = 0.25 * rawVelPend + 0.75 * omegaPendFiltered;

    // Current pendulum angle (for safety check)
    float pendAngle = cPend * TICK_TO_RAD * PEND_SIGN;

    // ── Safety cutoff: stop motor if pendulum outside stabilisation zone ──
    if (fabs(pendAngle) > SAFE_PEND_ANGLE) {
      safetyCutoff = true;
      stopMotor();
    }

    // Build and send binary state packet
    StatePacket pkt;
    pkt.thetaPend = pendAngle;
    pkt.thetaArm  = cArm * TICK_TO_RAD * ARM_SIGN;
    pkt.omegaPend = omegaPendFiltered  * PEND_SIGN;
    pkt.omegaArm  = omegaArmFiltered   * ARM_SIGN;

    // Send the 2-byte Sync Word (0xABCD)
    // Arduino is little-endian, so sending this uint16_t puts 0xCD then 0xAB on the wire
    uint16_t syncWord = 0xABCD;
    Serial.write((byte*)&syncWord, sizeof(syncWord));
    
    // Send the 16-byte payload
    Serial.write((byte*)&pkt, sizeof(pkt));

    lastCountArm  = cArm;
    lastCountPend = cPend;
  }

  // ── 2. Non-blocking serial receive (voltage command, 4 bytes) ────────────
  if (Serial.available() >= 4) {
    // Discard stale packets: if multiple commands queued, skip to latest
    while (Serial.available() >= 8) {
      // Throw away one stale 4-byte command
      uint8_t trash[4];
      Serial.readBytes(trash, 4);
    }

    float voltageCommand;
    Serial.readBytes((byte*)&voltageCommand, 4);
    lastCommandTime = millis();

    if (!safetyCutoff) {
      applyVoltage(voltageCommand);
    }
  }

  // ── 3. Watchdog: stop motor if Python stops sending ──────────────────────
  if (millis() - lastCommandTime > SAFETY_TIMEOUT) {
    stopMotor();
  }
}

// ── Motor Control ───────────────────────────────────────────────────────────
void applyVoltage(float volts) {
  volts *= MOTOR_SIGN;

  // 1. Deadband Compensation (Using your 0.5V logic)
  if (volts > THRESH) { 
    volts = volts + DEADZONE;
  } else if (volts < -THRESH) {
    volts = volts - DEADZONE;
  } else {
    volts = 0.0; // Ignore sensor noise near zero to prevent motor buzzing
  }

  // 2. Hard voltage clamp
  volts = constrain(volts, -SUPPLY_VOLTAGE, SUPPLY_VOLTAGE);

  // 3. Duty cycle
  float duty = constrain(volts / SUPPLY_VOLTAGE, -1.0, 1.0);

  // 4. Convert to PWM (0-255)
  int pwm = (int)(fabs(duty) * 255.0 + 0.5);   // round instead of truncate
  
  digitalWrite(motArmDir, (duty >= 0.0) ? HIGH : LOW);
  analogWrite(motArmPWM, pwm);
}

void stopMotor() {
  analogWrite(motArmPWM, 0);
}

// ── Encoder ISRs (free-running, no wrap) ────────────────────────────────────
void isrArmA()  { (digitalRead(encArmPinA) != digitalRead(encArmPinB))  ? countArm++  : countArm--;  }
void isrArmB()  { (digitalRead(encArmPinA) == digitalRead(encArmPinB))  ? countArm++  : countArm--;  }
void isrPendA() { (digitalRead(encPendPinA) != digitalRead(encPendPinB)) ? countPend-- : countPend++; }
void isrPendB() { (digitalRead(encPendPinA) == digitalRead(encPendPinB)) ? countPend-- : countPend++; }