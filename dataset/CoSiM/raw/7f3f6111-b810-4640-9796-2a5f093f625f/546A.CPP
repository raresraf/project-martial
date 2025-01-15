#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  ll k, n, w;
  cin >> k >> n >> w;
  int ans = k * w * (w + 1) / 2 - n;
  cout << max(ans, 0);
  return 0;
}