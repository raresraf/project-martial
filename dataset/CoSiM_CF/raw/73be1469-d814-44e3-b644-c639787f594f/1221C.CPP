#include <iostream>
typedef long long ll;
using namespace std;
ll q, c, m, x;

bool check(ll teams) {
  if ((teams > c) || (teams > m))
    return false;
  if (3 * teams > c + m + x)
    return false;
  return true;
}

ll search_teams(ll a, ll b) {
  ll mid = (a + b) / 2;
  // cout << "search_teams(" << a << ", " << b << ") mid = " << mid << endl;
  if (check(mid)) {
    if (!check(mid + 1))
      return mid;
    return search_teams(mid + 1, b);
  }
  return search_teams(a, mid - 1);
}
int main() {
  cin >> q;
  while (q--) {
    cin >> c >> m >> x;
    ll teams = (c + m + x) / 3;
    teams = search_teams(0, teams);
    cout << teams << endl;
  }
  return 0;
}