#include <iostream>

using namespace std;

int main() {
  int moveCount = 0;
  for (int i = 0; i < 5; i++) {
    bool hasOne = false;
    int a, b, c, d, e;
    cin >> a >> b >> c >> d >> e;
    if (a == 1 || b == 1 || c == 1 || d == 1 || e == 1) {
      hasOne = true;
    }
    if (a == 1 || e == 1) {
      moveCount += 2;
    }
    if (b == 1 || d == 1) {
      moveCount += 1;
    }

    if (hasOne == true && (i == 0 || i == 4)) {
      moveCount += 2;
    }

    if (hasOne == true && (i == 1 || i == 3)) {
      moveCount += 1;
    }
  }
  cout << moveCount << "\n";
  return 0;
}