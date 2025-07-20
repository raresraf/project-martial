#include <iostream>

using namespace std;

int main() {
  int t, ingreso, salida, max = 0, cantidad_pasajeros = 0;
  cin >> t;
  while (t--) {
    cin >> salida >> ingreso;
    cantidad_pasajeros += (ingreso - salida);
    if ((cantidad_pasajeros) > max) {
      max = cantidad_pasajeros;
    }
  }
  cout << max;
  return 0;
}
