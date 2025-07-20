#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int row = 0, col = 0;
  int tmp;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> tmp;
      if (tmp == 1) {
        row = j + 1;
        col = i + 1;
      }
    }
  }
  int steps = abs(row - 3) + abs(col - 3);
  cout << steps << endl;
}
