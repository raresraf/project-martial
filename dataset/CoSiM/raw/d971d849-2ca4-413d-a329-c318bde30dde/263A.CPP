#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, j, x, y, arr[15][15];
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }
  cout << abs(x - 3) + abs(y - 3) << endl;
  return 0;
}
