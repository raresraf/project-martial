#include <bits/stdc++.h>
#include <string>
using namespace std;
// char b[10010];

int main() {
  int n, i = 0, j, u = 0, l, f = 0, k, ra = 0, c = 0, r = 0, s = 0, sum = 0, m;
  char a[10010], b[10010];
  double x, y, z;

  cin >> a;

  n = strlen(a);
  for (i = 0; i < n; i++) {
    if (a[i] == '4' || a[i] == '7')
      f++;
  }
  if (f == 0)
    s = 1;
  while (f > 0) {
    if (f % 10 != 4 && f % 10 != 7)
      s = 1;
    f /= 10;
  }
  if (s == 1)
    cout << "NO\n";
  else
    cout << "YES\n";

  return 0;
}