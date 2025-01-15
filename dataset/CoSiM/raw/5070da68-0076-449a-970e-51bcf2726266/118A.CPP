#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string palavra;
  int contador;

  cin >> palavra;
  string letras = "aeiouy";
  string nova_palavra = "";

  for (int i = 0; i < palavra.size(); i++) {

    palavra[i] = (tolower(palavra[i]));

    contador = 0;
    for (int j = 0; j < letras.size(); j++) {
      if (palavra[i] == letras[j]) {
        contador += 1;
      }
    }
    if (contador == 0) {
      nova_palavra += ".";
      nova_palavra += palavra[i];
    }
  }

  cout << nova_palavra << endl;
  return 0;
}