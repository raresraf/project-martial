#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, m, i, j, q, y, aa, bb, f = 0, s = 0, k, min = 1000000010, r, t, u = -1;
  char a[10010], b[10010];
  ;

  cin >> n >> a;
  r = strlen(a);
  for (i = 0; i < r; i++) {
    if (a[i] == a[i + 1])
      f++;
  }
  cout << f << endl;

  return 0;
}