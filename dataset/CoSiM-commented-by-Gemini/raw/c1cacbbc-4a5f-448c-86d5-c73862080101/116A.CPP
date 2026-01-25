#include <stdio.h>

using namespace std;

int main() {
  int n, a, b, minTram = 0, totalPassengers = 0;
  scanf("%d", &n);
  while (n--) {
    scanf("%d %d", &a, &b);
    totalPassengers -= a;
    totalPassengers += b;
    if (totalPassengers > minTram)
      minTram = totalPassengers;
  }
  printf("%d\n", minTram);
  return 0;
}
/*
    Es un problema facil, nos piden que hallemos la minima longitud del tramo
   para que el tren no se sature. En realidad nos piden hallar es la maxima
   capacidad de pasajeros que puede obtener el tren en los datos de entrada y
   eso puede ser facilmente calculado sumando y restando los pasajeros que
   entran y salen por cada iteracion para saber cuantos pasajeros hay en el tren
   por cada iteracion.

*/