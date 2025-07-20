#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  int m, n;
  bool found = false;

  for (int i = 0; i < 5; i++) {
    int c1, c2, c3, c4, c5;
    cin >> c1 >> c2 >> c3 >> c4 >> c5;

    if (found) {
      continue;
    }

    if (c1 == 1) {
      m = i;
      n = 0;
      found = true;
    }

    if (c2 == 1) {
      m = i;
      n = 1;
      found = true;
    }

    if (c3 == 1) {
      m = i;
      n = 2;
      found = true;
    }

    if (c4 == 1) {
      m = i;
      n = 3;
      found = true;
    }

    if (c5 == 1) {
      m = i;
      n = 4;
      found = true;
    }
  }

  cout << abs(m - 2) + abs(n - 2) << endl;

  return 0;
}