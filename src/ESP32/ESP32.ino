#include <WiFi.h>
#include <HTTPClient.h>

#define GREEN_D2 2  // Pin D2
#define RED_D4 4    // Pin D4

const char* ssid = "MSI 7758";
const char* password = "744H26vg";
const char* backendUrl = "http://192.168.1.10:8000/emotion";

void setup() {
  Serial.begin(115200);
  pinMode(GREEN_D2, OUTPUT);
  pinMode(RED_D4, OUTPUT);
  digitalWrite(GREEN_D2, LOW);
  digitalWrite(RED_D4, LOW);
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(backendUrl);
    
    int httpResponseCode = http.GET();
    
    if (httpResponseCode == HTTP_CODE_OK) {
      String payload = http.getString();
      Serial.println("Raw response: " + payload);

      // Manual JSON parsing
      int emotionValue = parseEmotionValue(payload);
      Serial.print("Parsed emotion value: ");
      Serial.println(emotionValue);
      
      // Control LEDs
      if (emotionValue == 3) {
        digitalWrite(GREEN_D2, HIGH);
        digitalWrite(RED_D4, LOW);
        Serial.println("Emotion 3 - GREEN ON");
      } else {
        digitalWrite(GREEN_D2, LOW);
        digitalWrite(RED_D4, HIGH);
        Serial.println("Other emotion - RED ON");
      }
    } else {
      Serial.print("HTTP Error: ");
      Serial.println(httpResponseCode);
    }
    
    http.end();
  } else {
    Serial.println("WiFi disconnected - attempting reconnect");
    WiFi.begin(ssid, password);
    delay(5000); // Wait before retrying
  }
  
  delay(2000);
}

// Simple JSON parser for {"emotion":X} format
int parseEmotionValue(String json) {
  return json.toInt(); // Convert to integer
}