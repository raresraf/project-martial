#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  ll total = w * (w + 1) / 2 * k;
  if (total <= n) {
    cout << 0;
  } else {
    cout << total - n;
  }
  return 0;
}