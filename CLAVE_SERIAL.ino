const int correctOrder[] = {1, 2, 3, 4}; // Orden específico deseado
int receivedOrder[4]; // Array para almacenar los números recibidos
int index = 0; // Índice para el array de números recibidos
void setup() 
{Serial.begin(9600); // Inicializa la comunicación serial a 9600 bps
 pinMode(13, OUTPUT); // Configura el pin 13 como salida
 digitalWrite(13, LOW); // Asegura que el pin 13 esté inicialmente en estado bajo
}
void loop()
{
  if (Serial.available() > 0) // Verifica si hay datos disponibles en el puerto serial
  {
    int receivedNumber = Serial.parseInt(); // Lee un número entero del puerto serial
    if (index < 4){ receivedOrder[index] = receivedNumber; index++; } // Almacena el número en el array
    if (index == 4) // Verifica si se han recibido 4 números
    { // Verifica si el orden de los números es el correcto
      bool isCorrect = true; for (int i = 0; i < 4; i++) { if (receivedOrder[i] != correctOrder[i]) { isCorrect = false; break; } }
      // Si el orden es correcto, genera la clave de acceso y manda un estado alto por el pin 13 durante 10 segundos y después se apaga
      if (isCorrect) { digitalWrite(13, HIGH); delay(10000); digitalWrite(13, LOW); } 
      else { digitalWrite(13, LOW); }// Asegura que el pin 13 esté en estado bajo si el orden es incorrecto
      index = 0; // Reinicia el índice para recibir nuevos números
    }
  }
}
