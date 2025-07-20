// Codeforces
// 263A - Beautiful matrix

#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int n;
  int x, y;

  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> n;
      if (n == 1) {
        x = i;
        y = j;
      }
    }
  }

  int nbrMoves = abs(3 - x) + abs(3 - y);

  cout << nbrMoves;

  return 0;
}
