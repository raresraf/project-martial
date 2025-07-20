#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#define rep(i, n) for (int i = 1; i <= n; i++)
#define debug(x) cout << #x << "=" << x << endl;
using namespace std;

int n, tot, x[500], y[500];

int main() {
  scanf("%d", &n);
  rep(i, n) scanf("%d%d", &x[i], &y[i]);
  rep(i, n) {
    bool l = 0, r = 0, u = 0, d = 0;
    rep(j, n) if (x[j] == x[i] && y[j] < y[i]) d = 1;
    else if (x[j] == x[i] && y[j] > y[i]) u = 1;
    else if (y[i] == y[j] && x[i] > x[j]) l = 1;
    else if (y[i] == y[j] && x[i] < x[j]) r = 1;
    if (l && r && u && d)
      tot++;
  }
  cout << tot << endl;
  return 0;
}
