#include <bits/stdc++.h>

using namespace std;

int petya(string cadena1, string cadena2);

int main() {
  string cadena1, cadena2;
  char s;

  cin >> cadena1;
  cin >> cadena2;

  cout << petya(cadena1, cadena2);

  return 0;
}

int petya(string cadena1, string cadena2) {
  int flag = 0;
  int subs = 0;

  for (int i = 0; i < cadena1.size(); ++i) {
    // ambas las convertimos en may sculas
    if (cadena1[i] > 90) {
      cadena1[i] -= 32;
    }
    if (cadena2[i] > 90) {
      cadena2[i] -= 32;
    }

    // hacemos la diferencia
    subs = cadena1[i] - cadena2[i];

    if (subs == 0)
      flag = 0;
    else if (subs >= -25 && subs < 0)
      return -1;
    else if (subs <= 25 && subs > 0)
      return 1;
  }

  return 0;
}