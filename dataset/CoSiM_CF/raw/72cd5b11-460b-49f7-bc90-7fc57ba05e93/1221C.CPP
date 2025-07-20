//  [amitdu6ey]
// g++ -std=c++11 -o2 -Wall filename.cpp -o filename
#include <bits/stdc++.h>
#define hell 1000000009
#define bug1(x) cout << "$ " << x << " $" << endl
#define bug2(x) cout << "% " << x << " %" << endl
#define bug3(x) cout << "# " << x << " #" << endl
#define ll long long
#define pb push_back
#define mp make_pair
#define tr(z) for (auto it = z.begin(); it != z.end(); it++)
#define rloop(i, a, b) for (long long i = a; i >= b; i--)
#define loop(i, a, b) for (long long i = a; i < b; i++)
#define vbool vector<bool>
#define vvbool vector<vector<bool>>
#define vchar vector<char>
#define vi vector<int>
#define vvi vector<vector<int>>
#define pll pair<long long, long long>
#define vll vector<long long>
#define vvl vector<vector<long long>>
#define ninf numeric_limits<long long>::min()
#define pinf numeric_limits<long long>::max()
#define comp(a, b) (abs(a - b) < 1e-9) // to compare doubles
using namespace std;

bool team_possible(ll c, ll m, ll x, ll ts) {
  if (ts > min(c, m)) {
    return false;
  }
  m -= ts;
  c -= ts;
  if (ts <= c + m + x) {
    return true;
  }
  return false;
}

void solve() {
  ll c, m, x;
  cin >> c >> m >> x;
  ll l = 0, r = 1e9, mid = (l + r) / 2;
  ll ans = 0;
  while (l <= r) {
    mid = (l + r) / 2;
    if (team_possible(c, m, x, mid)) {
      l = mid + 1;
      ans = mid;
    } else {
      r = mid - 1;
    }
  }
  cout << ans << "\n";
}

int main() {
  ios_base::sync_with_stdio(0);
  cout.tie(0);
  cin.tie(0);
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  ll test_cases = 1;
  cin >> test_cases;
  while (test_cases--) {
    solve();
  }
  return 0;
}