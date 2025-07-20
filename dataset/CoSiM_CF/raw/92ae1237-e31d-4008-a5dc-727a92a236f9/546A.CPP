#include <bits/stdc++.h>
using namespace std;
int main() {

  long long k, n, w;
  cin >> k >> n >> w;

  long long ans;
  ans = k * w * (w + 1);
  ans /= 2;
  ans -= n;

  if (ans > 0)
    cout << ans;
  else
    cout << "0";

  return 0;
}
