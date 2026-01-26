#include <bits/stdc++.h>
using namespace std;
main() {
  int n;
  cin >> n;
  int a[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  int k = sum / 2;
  int coin = 0;
  sum = 0;
  int i = n - 1;
  sort(a, a + n);

  while (1) {
    coin++;
    sum += a[i];
    if (sum > k) {
      cout << coin;
      break;
    }
    i--;
  }
}
