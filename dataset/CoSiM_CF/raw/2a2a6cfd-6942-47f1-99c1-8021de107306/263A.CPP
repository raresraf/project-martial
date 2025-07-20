#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int numbers[5][5];

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> numbers[i][j];
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (numbers[i][j] == 1) {
        int x = i;
        int y = j;
        x = abs(x - 2);
        y = abs(y - 2);
        cout << "" << x + y;
      }
    }
  }

  return 0;
}