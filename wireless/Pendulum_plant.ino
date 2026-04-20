#include <WiFi.h>
#include <esp_now.h>

/* ===================== ESP-NOW CONFIG ===================== */

// Controller MAC
uint8_t broadcastAddress[] = {0xCC, 0xDB, 0xA7, 0x61, 0xF8, 0x78};

// Sampling time
const unsigned long Ts = 5;
unsigned long last_sample_time = 0;

/* ===================== PWM CONFIG ===================== */

const int motArmPWM = 5;
const int motArmDir = 4;

const int pwmFreq = 2000;
const int pwmResolution = 8;

const float SUPPLY = 12.0;

/* ===================== ENCODER CONFIG ===================== */

const int armPinA = 32;
const int armPinB = 33;
const int pendPinA = 19;
const int pendPinB = 18;

volatile long armCount = 0;
volatile long pendCount = 0;

const int totalTicks = 2000;
const int halfTicks = 1000;
const float TICK_TO_RAD = 3.1415926535 / 1000.0;

long armlastCount = 0;
long pendlastCount = 0;

float filarmVel = 0;
float filpendVel = 0;
const float alpha = 0.8;

/* ===================== CONTROL ===================== */

float current_control = 0;

/* ===================== PACKETS ===================== */

enum PacketType {
  STATE_PACKET = 1,
  CONTROL_PACKET = 2
};

typedef struct {
  uint8_t type;
  unsigned long timestamp;
  float pendulum_angle;
  float pendulum_velocity;
  float arm_angle;
  float arm_velocity;
} StatePacket;

typedef struct {
  uint8_t type;
  unsigned long timestamp;
  float control_input;
} ControlPacket;

StatePacket outgoing_state_packet;
ControlPacket incoming_control_packet;

esp_now_peer_info_t peerInfo;

/* ===================== ENCODER ISR ===================== */

void IRAM_ATTR readEncoder1A() {
  (digitalRead(armPinA) != digitalRead(armPinB)) ? armCount++ : armCount--;
}
void IRAM_ATTR readEncoder1B() {
  (digitalRead(armPinA) == digitalRead(armPinB)) ? armCount++ : armCount--;
}
void IRAM_ATTR readEncoder2A() {
  (digitalRead(pendPinA) != digitalRead(pendPinB)) ? pendCount++ : pendCount--;
}
void IRAM_ATTR readEncoder2B() {
  (digitalRead(pendPinA) == digitalRead(pendPinB)) ? pendCount++ : pendCount--;
}

/* ===================== MOTOR ===================== */

void applyVoltage(float volts) {

  if (volts > 0.05) volts += 1.41;
  else if (volts < -0.05) volts -= 1.41;
  else volts = 0;

  float duty = volts / SUPPLY;

  if (duty > 1.0) duty = 1.0;
  if (duty < -1.0) duty = -1.0;

  int pwm = abs(duty) * 255;

  digitalWrite(motArmDir, (duty > 0) ? HIGH : LOW);
  ledcWrite(motArmPWM, pwm);
}

void stopMotor() {
  ledcWrite(motArmPWM, 0);
}

/* ===================== ESP-NOW CALLBACK ===================== */

void OnReceiveData(const esp_now_recv_info *recv_info,
                   const uint8_t* incomingData,
                   int len)
{
  if (len == sizeof(ControlPacket)) {

    memcpy(&incoming_control_packet, incomingData, sizeof(ControlPacket));

    if (incoming_control_packet.type == CONTROL_PACKET) {

      current_control = incoming_control_packet.control_input;

      // ✅ APPLY IMMEDIATELY (only if within safe angle)
      long pendCurrent;
      noInterrupts();
      pendCurrent = pendCount;
      interrupts();

      float pend = pendCurrent * TICK_TO_RAD;

      if (abs(pend) < 0.35) {
        applyVoltage(current_control);
      } else {
        stopMotor();
      }
    }
  }
}

/* ===================== SETUP ===================== */

void setup() {

  Serial.begin(115200);

  pinMode(armPinA, INPUT_PULLUP);
  pinMode(armPinB, INPUT_PULLUP);
  pinMode(pendPinA, INPUT_PULLUP);
  pinMode(pendPinB, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(armPinA), readEncoder1A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(armPinB), readEncoder1B, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pendPinA), readEncoder2A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pendPinB), readEncoder2B, CHANGE);

  ledcAttach(motArmPWM, pwmFreq, pwmResolution);
  pinMode(motArmDir, OUTPUT);

  stopMotor();

  // ESP-NOW setup
  WiFi.mode(WIFI_STA);
  esp_now_init();
  esp_now_register_recv_cb(OnReceiveData);

  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);
}

/* ===================== MAIN LOOP ===================== */

void loop() {

  unsigned long now = millis();

  if (now - last_sample_time >= Ts) {

    float dt = (now - last_sample_time) / 1000.0;
    last_sample_time = now;

    long armCurrent, pendCurrent;

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

    filarmVel = alpha * armVel + (1 - alpha) * filarmVel;
    filpendVel = alpha * pendVel + (1 - alpha) * filpendVel;

    float arm = armCurrent * TICK_TO_RAD;
    float pend = pendCurrent * TICK_TO_RAD;

    armlastCount = armCurrent;
    pendlastCount = pendCurrent;

    // Send state
    outgoing_state_packet.type = STATE_PACKET;
    outgoing_state_packet.timestamp = micros();
    outgoing_state_packet.arm_angle = arm;
    outgoing_state_packet.arm_velocity = filarmVel;
    outgoing_state_packet.pendulum_angle = pend;
    outgoing_state_packet.pendulum_velocity = filpendVel;

    esp_now_send(broadcastAddress,
                 (uint8_t*)&outgoing_state_packet,
                 sizeof(StatePacket));

    // ✅ ALSO APPLY LAST CONTROL (ZOH)
    if (abs(pend) < 0.35) {
      applyVoltage(current_control);
    } else {
      stopMotor();
    }
  }
}