#include <bits/stdc++.h>

using namespace std;

int main() {

  int n[5][5], m[1][1], inrow, incol, cnt = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> n[i][j];
      if (n[i][j] == 1) {
        inrow = i;
        incol = j;
      }
    }
  }

  for (int j = 0; j < 25; j++) {
    if (incol > 2) {
      incol--;
      cnt++;
    } else if (incol < 2) {
      incol++;
      cnt++;
    } else if (incol == 2 && inrow > 2) {
      inrow--;
      cnt++;
    } else if (incol == 2 && inrow < 2) {
      inrow++;
      cnt++;
    } else if (incol == 2 && inrow == 2) {
      cout << cnt << "\n";
      return 0;
    }
  }

  return 0;
}
