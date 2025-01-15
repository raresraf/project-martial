#include <bits/stdc++.h>
#define fast                                                                   \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(NULL);                                                               \
  cout.tie(NULL)
#define ll long long int
#define ld long double
using namespace std;
const int N = 1e6 + 5;
const int MOD = 1e9 + 7;

ll l[N], r[N];
int main() {
  fast;
  ll n;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    cin >> l[i] >> r[i];
  }
  ll x;
  cin >> x;
  for (int i = 1; i <= n; i++) {
    if (x <= r[i]) {
      return cout << n - i + 1, 0;
    }
  }

  return 0;
}
