#include <bits/stdc++.h>
#include <string>
using namespace std;
int b[10010];

int main() {
  int n, m, i, j, q, y, aa, bb, f = 0, min = 10000010, t = 1, r, s = 0, u,
                                max = 0;
  double div;
  int a[10010];
  cin >> n;
  for (i = 1; i <= n; i++) {
    cin >> a[i];
    s += a[i];
  }
  sort(a, a + n + 1);
  s = s / 2;
  s++;
  for (i = n; i > 0; i--) {
    if (s > 0) {

      s -= a[i];
      f++;
    }
    // cout << a[i];
  }

  cout << f << endl;

  return 0;
}