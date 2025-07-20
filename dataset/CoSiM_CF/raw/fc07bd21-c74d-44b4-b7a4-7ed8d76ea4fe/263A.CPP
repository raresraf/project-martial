#include <bits/stdc++.h>
using namespace std;
int main() {
  int a[5][5];
  int i, j, sum = 0;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        if (i >= 2)
          sum = i - 2;
        else if (i < 2)
          sum = 2 - i;
        if (j >= 2)
          sum += j - 2;
        else if (j < 2)
          sum += 2 - j;
      }
    }
  }
  cout << sum << endl;
}
