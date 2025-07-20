#include <bits/stdc++.h>

using namespace std;

int main() {
  long long k, w, n;
  cin >> k >> w >> n;
  long long ans = n * (n + 1) / 2 * k;
  if (ans <= w)
    cout << 0 << endl;
  else
    cout << ans - w << endl;
  return 0;
}
