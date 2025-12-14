#include <iostream>
using namespace std;
int main() {
  int row, col;
  int num;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> num;
      if (num == 1) {
        row = i;
        col = j;
      }
    }
  }
  int steps = (row > 2 ? row - 2 : 2 - row) + (col > 2 ? col - 2 : 2 - col);
  cout << steps;
}