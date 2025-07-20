#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int i, j, moves, number, row, column;
  int beautifulIndex = 2;

  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++) {
      cin >> number;
      if (number) {
        row = i;
        column = j;
      }
    }

  moves = abs(beautifulIndex - row) + abs(beautifulIndex - column);
  cout << moves << endl;

  return 0;
}