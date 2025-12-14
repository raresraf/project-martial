#include <bits/stdc++.h>

using namespace std;

int main() {

  int c = 0;

  int k, n, a[51], i;
  cin >> n >> k;

  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (i = 0; i < n; i++) {
    if (a[i] > 0 && a[i] >= a[k - 1]) {
      c++;
    } else {
      break;
    }
  }
  cout << c;
}
