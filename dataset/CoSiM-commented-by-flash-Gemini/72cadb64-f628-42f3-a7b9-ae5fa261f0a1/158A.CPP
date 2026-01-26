#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, a[105], b, cn = 0;
  cin >> n >> k;
  for (int i = 1; i <= n; ++i) {
    cin >> a[i];
  }
  b = a[k];

  for (int i = 1; i <= n; ++i) {
    if (a[i] >= b && a[i] > 0)
      ++cn;
  }
  cout << cn << endl;
}
