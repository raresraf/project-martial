#include <iostream>
using namespace std;

int main() {
  int N, T;
  cin >> N >> T;
  char fila[N];
  cin >> fila;

  for (int i = 0; i < T; i++) {
    for (int j = 1; j < N; j++) {
      if (fila[j - 1] == 'B' && fila[j] == 'G') {
        char aux = fila[j];
        fila[j] = fila[j - 1];
        fila[j - 1] = aux;
        j++;
      }
    }
  }
  cout << fila << endl;

  return 0;
}