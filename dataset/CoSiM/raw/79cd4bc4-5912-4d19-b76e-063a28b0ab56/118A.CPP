#include <cctype>
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

/* 	borrar vocales aeiouy
        agregar un punto antes de una consonante
        reemplazar todas las consonantes mayusculas por minusculas

        -->
        esVocal?
                s , --> borrar
                sino(es consonante) -->
                        agregar un punto antes de esta letra
                        esMayuscula?
                                s  -->	hacerlaMinuscula.

 _________________________

        Programa debe retornar el output como UN s lo string,
         resultando luego de procesar el string

         IP: primera linea, string con longitud de 1 a 100
         OP: imprimir el string resultante, se debe garantizar que este string
 no est  vac o
*/

int main() {
  string in, out = "";
  bool isVoel;
  char vowels[] = {'A', 'a', 'E', 'e', 'O', 'o', 'U', 'u', 'I', 'i', 'Y', 'y'};
  getline(cin, in);
  for (int i = 0; i < in.size(); i++) {
    if (in[i] <= 'Z' && in[i] >= 'A')
      in[i] = in[i] - ('Z' - 'z');
    isVoel = false;
    for (int j = 0; j < sizeof(vowels); j++) {
      if (in[i] == vowels[j]) {
        isVoel = true;
      }
    }
    if (isVoel == false)
      out = out + '.' + in[i];
  }
  cout << out << endl;

  return 0;
}