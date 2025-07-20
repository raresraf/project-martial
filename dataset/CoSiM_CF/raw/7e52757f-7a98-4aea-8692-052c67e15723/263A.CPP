#include <bits/stdc++.h>
using namespace std;

int main() {

  int ara[6][6], nmoves;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> ara[i][j];
    }
  }
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      if (ara[i][j] == 1) {
        // cout << i << " " << j << endl;
        nmoves = abs(3 - i) + abs(3 - j);
        break;
      }
    }
  }
  cout << nmoves << endl;

  return 0;
}
