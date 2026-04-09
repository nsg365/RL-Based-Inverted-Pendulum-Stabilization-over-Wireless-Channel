#include <WiFi.h>
#include <esp_now.h>

// MAC Address of Controller
uint8_t broadcastAddress[] = {0xCC, 0xDB, 0xA7, 0x61, 0xF8, 0x78};

// Sampling time
unsigned long last_sample_time = 0;
const unsigned long Ts = 15;  // ms (200 Hz)

// Global variables
float current_control = 0.0;
bool control_received = false;

// Delay statistics
unsigned long max_delay = 0;
unsigned long delay_sum = 0;
unsigned long packet_count = 0;
unsigned long last_print_time = 0;

// State of Plant
enum PlantState {
  SAMPLE,
  SEND_STATE,
  WAIT_CONTROL
};

PlantState plant_state = SAMPLE;

// Packet Types
enum PacketType {
  STATE_PACKET = 1,
  CONTROL_PACKET = 2
};

// State Packet
typedef struct {
  uint8_t type;
  unsigned long timestamp;  // microsecond timestamp
  float pendulum_angle;
  float pendulum_velocity;
  float arm_angle;
  float arm_velocity;
} StatePacket;

// Control Packet
typedef struct {
  uint8_t type;
  unsigned long timestamp;  // echoed timestamp
  float control_input;
} ControlPacket;

ControlPacket incoming_control_packet;
StatePacket outgoing_state_packet;

// Peer Info
esp_now_peer_info_t peerInfo;

// ================= RECEIVE CALLBACK =================
void OnReceiveData(const esp_now_recv_info *recv_info,
                   const uint8_t* incomingData,
                   int len)
{
  memcpy(&incoming_control_packet, incomingData, sizeof(ControlPacket));

  if (incoming_control_packet.type == CONTROL_PACKET)
  {
    unsigned long now = micros();
    unsigned long delay = now - incoming_control_packet.timestamp;

    // Update statistics
    packet_count++;
    delay_sum += delay;

    if (delay > max_delay)
      max_delay = delay;

    current_control = incoming_control_packet.control_input;
    control_received = true;
  }
}

// ================= SETUP =================
void setup()
{
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(OnReceiveData);

  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }
}

// ================= LOOP =================
void loop()
{
  // Sampling timer
  if (millis() - last_sample_time >= Ts) {
    last_sample_time = millis();
    plant_state = SAMPLE;
  }

  // SAMPLE STATE
  if (plant_state == SAMPLE) {

    outgoing_state_packet.type = STATE_PACKET;
    outgoing_state_packet.timestamp = micros();  // microsecond timestamp

    // Dummy sensor values
    outgoing_state_packet.arm_angle = 5.4;
    outgoing_state_packet.arm_velocity = 4.9;
    outgoing_state_packet.pendulum_angle = 10.2;
    outgoing_state_packet.pendulum_velocity = 2;

    plant_state = SEND_STATE;
  }

  // SEND STATE
  if (plant_state == SEND_STATE) {

    esp_now_send(broadcastAddress,
                 (uint8_t *)&outgoing_state_packet,
                 sizeof(StatePacket));

    plant_state = WAIT_CONTROL;
  }

  // WAIT CONTROL
  if (plant_state == WAIT_CONTROL) {

    if (control_received) {

      // Apply control (dummy print removed to avoid spam)
      control_received = false;
    }
  }

  // ===== Print Delay Statistics Once Per Second =====
  if (millis() - last_print_time > 1000) {

    Serial.println("------ Delay Stats (1 sec) ------");

    if (packet_count > 0) {
      unsigned long avg_delay = delay_sum / packet_count;

      Serial.print("Packets received: ");
      Serial.println(packet_count);

      Serial.print("Average RTT (us): ");
      Serial.println(avg_delay);

      Serial.print("Max RTT (us): ");
      Serial.println(max_delay);
    }
    else {
      Serial.println("No packets received");
    }

    Serial.println("----------------------------------");

    // Reset statistics
    packet_count = 0;
    delay_sum = 0;
    max_delay = 0;

    last_print_time = millis();
  }
}