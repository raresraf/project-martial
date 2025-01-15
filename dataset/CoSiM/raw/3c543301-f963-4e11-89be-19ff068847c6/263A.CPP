#include <iostream>
using namespace std;

int main() {
  int n = 25;
  auto row = 0;
  auto col = 0;
  for (int j = n; j != 0; j--) {
    int i;
    cin >> i;

    if (j % 5 == 0) {
      row++;
      col = 0;
    }
    col++;

    if (i == 1) {
      cout << (abs(col - 3) + abs(row - 3));
      break;
    }
  }
}