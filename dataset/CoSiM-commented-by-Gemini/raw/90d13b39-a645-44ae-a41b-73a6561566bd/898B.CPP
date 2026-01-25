#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  ll n, a, b;
  cin >> n >> a >> b;
  for (int i = 0; i <= n; i++) {
    ll j = (n - a * i) / b;
    if (a * i + b * j == n && j >= 0) {
      cout << "YES" << endl;
      cout << i << " " << j << endl;
      return 0;
    }
  }
  cout << "NO" << endl;
  return 0;
}
