#include <iostream>
using namespace std;

int main() {
  int grid[5][5], r, c;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> grid[i][j];
    }
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (grid[i][j] == 1) {
        r = i;
        c = j;
      }
    }
  }
  cout << abs(2 - r) + abs(2 - c) << endl;
  return 0;
}