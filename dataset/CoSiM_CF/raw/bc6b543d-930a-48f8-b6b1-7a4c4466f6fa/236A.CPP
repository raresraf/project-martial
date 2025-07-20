#include <iostream>

using namespace std;

int main() {
  string nombre;
  bool letras[26] = {0};
  int contador = 0;
  cin >> nombre;
  for (int i = 0; i < nombre.size(); i++) {
    if (!letras[nombre[i] - 97])
      letras[nombre[i] - 97] = 1;
  }
  for (int i = 0; i < 26; i++) {
    if (letras[i])
      contador++;
  }
  (contador % 2 == 0) ? cout << "CHAT WITH HER!" : cout << "IGNORE HIM!";
}
