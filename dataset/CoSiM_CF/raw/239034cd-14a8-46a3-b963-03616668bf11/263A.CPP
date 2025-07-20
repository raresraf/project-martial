#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;
int main() {
  int array[5][5];
  int moves = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> array[i][j];
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (array[i][j] == 1) {
        moves += (abs((i + 1) - 3));
        moves += (abs((j + 1) - 3));
      }
    }
  }

  cout << moves;
  return 0;
}
