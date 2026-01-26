#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int col_1, col_2, col_3, col_4, col_5;

  for (int row = 1; row < 6; row++) {
    cin >> col_1 >> col_2 >> col_3 >> col_4 >> col_5;
    if (col_1 == 1) {
      cout << 2 + abs(row - 3);
      break;
    } else if (col_2 == 1) {
      cout << 1 + abs(row - 3);
      break;
    } else if (col_3 == 1) {
      cout << abs(row - 3);
      break;
    } else if (col_4 == 1) {
      cout << 1 + abs(row - 3);
      break;
    } else if (col_5 == 1) {
      cout << 2 + abs(row - 3);
      break;
    } else {
      continue;
    }
  }
  return 0;
}