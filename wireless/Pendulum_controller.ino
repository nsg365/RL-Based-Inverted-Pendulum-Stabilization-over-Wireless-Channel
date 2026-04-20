#include <WiFi.h>
#include <esp_now.h>

/* ================= CONFIG ================= */

uint8_t plantAddress[] = {0xCC, 0xDB, 0xA7, 0x62, 0xB4, 0x2C};

// LQR Gains
float k1 = -4.4681;   // arm angle
float k2 = -1.6210;   // arm velocity
float k3 = -39.5037;  // pendulum angle
float k4 = -5.0402;   // pendulum velocity

/* ================= PACKETS ================= */

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

volatile bool state_received = false;
StatePacket incoming_state;
ControlPacket outgoing_control;

esp_now_peer_info_t peerInfo;

/* ================= CALLBACK ================= */

void OnReceiveData(const esp_now_recv_info *info,
                   const uint8_t* data,
                   int len)
{
  if (len == sizeof(StatePacket)) {
    memcpy(&incoming_state, data, sizeof(StatePacket));
    if (incoming_state.type == STATE_PACKET) {
      state_received = true;
    }
  }
}

/* ================= SETUP ================= */

void setup() {

  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  esp_now_init();
  esp_now_register_recv_cb(OnReceiveData);

  memcpy(peerInfo.peer_addr, plantAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  // Gain tuning (same as your original working one)
  k1 *= 0.75;
  k2 *= 0.5;
  k3 *= 1.55;
  k4 *= 0.4;
}

/* ================= LOOP ================= */

void loop() {

  if (state_received) {

    noInterrupts();
    StatePacket s = incoming_state;
    state_received = false;
    interrupts();

    /* ===== STATE MAPPING (CRITICAL) ===== */

    float arm     = s.arm_angle;
    float armVel  = s.arm_velocity;
    float pend    = s.pendulum_angle;
    float pendVel = s.pendulum_velocity;

    /* ===== SIGN FIX (if needed) ===== */
    // 👉 Uncomment ONLY if direction is wrong

    // pend = -pend;
    // pendVel = -pendVel;

    // arm = -arm;
    // armVel = -armVel;

    /* ===== CONTROL LAW (same as original) ===== */

    float u = -(k1 * arm +
                k2 * armVel +
                k3 * pend +
                k4 * pendVel);

    /* ===== SEND CONTROL ===== */

    outgoing_control.type = CONTROL_PACKET;
    outgoing_control.timestamp = s.timestamp;
    outgoing_control.control_input = u;

    esp_now_send(plantAddress,
                 (uint8_t*)&outgoing_control,
                 sizeof(ControlPacket));

    /* ===== DEBUG ===== */

    Serial.print("Pend: "); Serial.print(pend, 3);
    Serial.print(" | Arm: "); Serial.print(arm, 3);
    Serial.print(" | U: "); Serial.println(u, 3);
  }
}