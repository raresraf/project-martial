

#include <bits/stdc++.h>

using namespace std;

int main() {
  int b, c;
  int a[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (a[i][j] != 0) {
        b = i;
        c = j;
        break;
      }
    }
  }
  cout << abs(b - 2) + abs(c - 2);

  return 0;
}
