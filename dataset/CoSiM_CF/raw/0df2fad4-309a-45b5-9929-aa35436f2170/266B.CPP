#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, m, i, j, q, y, aa, bb, f = 0, s = 0, k, min = 1000000010, r, t, u = -1;
  char a[10010], b[10010];
  int check = 1;
  cin >> n >> t;
  cin >> a;
  while (t--) {
    for (i = n - 1; i >= 0; i--) {
      if (a[i] == 'G' && a[i - 1] == 'B') {
        a[i - 1] = 'G';
        a[i] = 'B';
        i--;
      }
    }
  }

  cout << a << endl;

  return 0;
}