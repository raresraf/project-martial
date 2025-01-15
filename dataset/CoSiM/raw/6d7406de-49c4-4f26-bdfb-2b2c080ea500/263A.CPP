#include <bits/stdc++.h>
using namespace std;
int main() {
  int ara[6][6], i, j, x, y;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> ara[i][j];

      if (ara[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }
  if (3 - x < 0)
    x = -1 * (3 - x);
  else
    x = 3 - x;
  if (3 - y < 0)
    y = -1 * (3 - y);
  else
    y = 3 - y;
  cout << x + y << endl;
}
