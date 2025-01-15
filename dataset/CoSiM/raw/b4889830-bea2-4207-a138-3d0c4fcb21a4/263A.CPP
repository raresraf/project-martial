#include <bits/stdc++.h>
using namespace std;
main() {

  int a[6][6];
  int l, k;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        l = i;
        k = j;
      }
    }
  }

  k = 3 - k;
  l = 3 - l;

  if (k < 0)
    k *= -1;
  if (l < 0)
    l *= -1;

  cout << k + l;
}
