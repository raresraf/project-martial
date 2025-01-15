#include <iostream>
using namespace std;

int main() {
  int a, b, cont = 0;
  cin >> a >> b;
  int aux, aux2;
  for (int i = 0; i < a; i++) {
    aux = aux2;
    cin >> aux2;
    if (!aux2 > 0)
      break;
    if ((i + 1) > b && aux2 < aux)
      break;
    cont++;
  }
  cout << cont << endl;
  return 0;
}
