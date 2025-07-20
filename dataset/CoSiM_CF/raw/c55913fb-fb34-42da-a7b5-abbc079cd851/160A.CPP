#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, i, total = 0, mine = 0;
  cin >> n;

  int a[n];
  for (i = 0; i < n; i++) {
    cin >> a[i];
    total += a[i];
  }

  sort(a, a + n);

  // start from biggest one
  while (i--) {
    mine += a[i];
    if (mine * 2 > total) {
      cout << n - i;
      return 0;
    }
  }
}