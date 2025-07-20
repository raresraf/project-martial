#include <bits/stdc++.h>
using namespace std;
#define ll long long
int main() {
  ll n, m, i, j, ans, k, w;
  while (cin >> n >> k >> w) {
    m = (w * (w + 1)) / 2;
    i = n * m;
    if (i > k) {
      ans = abs(i - k);
      cout << ans << endl;
    } else
      cout << "0" << endl;
  }
}
