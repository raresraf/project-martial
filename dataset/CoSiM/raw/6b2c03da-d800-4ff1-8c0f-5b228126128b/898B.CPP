#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  ll n = 1, q, i = 0, j = 0, a, b;
  cin >> n >> a >> b;
  bool s = true;
  for (i = 0; i * a < n; ++i) {
    ll m = n - i * a;
    if (m % b == 0) {
      s = false;
      cout << "YES\n";
      cout << i << " " << m / b << "\n";
      break;
    }
  }
  if (s)
    for (j = 0; j * b < n; ++j) {
      ll m = n - j * b;
      if (m % a == 0) {
        s = false;
        cout << "YES\n";
        cout << m / a << " " << j << "\n";
        break;
      }
    }
  if (s)
    cout << "NO\n";
  return 0;
}