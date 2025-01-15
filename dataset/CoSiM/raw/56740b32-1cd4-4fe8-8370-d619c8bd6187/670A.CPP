#include <bits/stdc++.h>

using namespace std;

#define int long long
#define ld long double
#define pb push_back
#define F first
#define S second
#define pii pair<int, int>
#define pi acos(-1)
#define X real()
#define Y imag()
#define point complex<ld>

const int MaxN = 2E5 + 10;
const ld eps = 1E-8;
const int INF = 1E9 + 7;
const int MOD = 1000 * 1000 * 1000 + 7;

int ghadr(int x) { return max(x, -x); }
int bpow(int n, int k, int delta = MOD) {
  int ans = 1;
  while (k) {
    if (k % 2)
      ans = (n * ans) % delta;
    n = (n * n) % delta;
    k /= 2;
  }
  return ans;
}

int32_t main() {
  ios::sync_with_stdio(false);
  cin.tie();
  cout.tie();
  cout << fixed << setprecision(12);
  int n;
  cin >> n;
  if (n % 7 == 0)
    cout << 2 * (n / 7) << " " << 2 * (n / 7);
  else if (n % 7 == 1)
    cout << 2 * (n / 7) << " " << 2 * (n / 7) + 1;
  else if (n % 7 == 6)
    cout << 2 * (n / 7) + 1 << " " << 2 * (n / 7) + 2;
  else
    cout << 2 * (n / 7) << " " << 2 * (n / 7) + 2;
  return 0;
}