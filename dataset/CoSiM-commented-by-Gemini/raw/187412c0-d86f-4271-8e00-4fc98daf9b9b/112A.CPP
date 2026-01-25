#include <iostream>

using namespace std;

int main() {
  string palabra1, palabra2;
  cin >> palabra1;
  cin >> palabra2;
  int resultado = 0;
  for (int i = 0; i < palabra1.size(); i++) {
    if (palabra1[i] < 92) {
      palabra1[i] += 32;
    }
    if (palabra2[i] < 92) {
      palabra2[i] += 32;
    }
  }
  if (palabra1 > palabra2)
    resultado = 1;
  else if (palabra1 < palabra2)
    resultado = -1;
  cout << resultado;
}
