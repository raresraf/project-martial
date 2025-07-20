#include <iostream>
#include <string>

using namespace std;

int main() {
  string sumar = "++", restar = "--";
  int n, resultado = 0;
  string operacion;
  cin >> n;
  while (n--) {
    cin >> operacion;
    if (operacion.find(sumar) < operacion.size()) {
      resultado++;
    } else if (operacion.find(restar) < operacion.size()) {
      resultado--;
    }
  }
  cout << resultado;
  return 0;
}
