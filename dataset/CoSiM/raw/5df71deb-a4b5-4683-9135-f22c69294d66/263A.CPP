#include <bits/stdc++.h>
using namespace std;
int main() {
  int t, i, j, pos[2];
  int x[5][5];
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> x[i][j];
      if (x[i][j] == 1) {
        pos[0] = i;
        pos[1] = j;
      }
    }
  }
  cout << fabs(2 - pos[0]) + fabs(2 - pos[1]);
}
