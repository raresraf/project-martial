#include <iostream>
#include <math.h>

using namespace std;

int main() {
  int matriz[5][5]{}, index[1][2]{};

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matriz[i][j];
      if (matriz[i][j] == 1) {
        index[1][1] = i;
        index[1][2] = j;
        j = 5;
        i = 5; // para sair do loop
      }
    }
  }
  cout << fabs(index[1][1] - 2) + fabs(index[1][2] - 2);

  return 0;
}