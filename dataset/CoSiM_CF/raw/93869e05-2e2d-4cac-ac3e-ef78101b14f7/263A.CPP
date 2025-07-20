#include <iostream>

using namespace std;

int main() {
  int a[5][5];
  int c, d;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        c = i;
        d = j;
      }
    }
  }

  int s = abs(d - 2) + abs(c - 2);
  cout << s;

  return 0;
}
