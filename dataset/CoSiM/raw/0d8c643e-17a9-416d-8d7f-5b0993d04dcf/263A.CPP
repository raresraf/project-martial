#include <bits/stdc++.h>

using namespace std;

int main() {
  int centro = 2;
  int pos_uno_fila = 0, pos_uno_columna = 0;
  int movimientos = 0;
  int e = 0;

  for (int i = 0; i < 25; i++) { // acceder de 5 en 5
    cin >> e;

    if (e == 1) {
      pos_uno_fila = i / 5;
      // cout << "fila: " << pos_uno_fila;
      pos_uno_columna = i - (pos_uno_fila * 5);
      // cout << "columna: " << pos_uno_columna;
      break;
    }
  }

  if (centro < pos_uno_fila)
    movimientos += pos_uno_fila - centro; // calculamos movimientos en filas
  else
    movimientos += centro - pos_uno_fila; // calculamos movimientos en filas

  if (centro < pos_uno_columna)
    movimientos +=
        pos_uno_columna - centro; // calculamos movimientos en columnas
  else
    movimientos +=
        centro - pos_uno_columna; // calculamos movimientos en columnas

  cout << movimientos << endl;

  return 0;
}