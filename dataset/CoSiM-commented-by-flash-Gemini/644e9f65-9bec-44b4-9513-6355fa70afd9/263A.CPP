/*
        Author   : Abhinav

        Modified : 24-08-2018 01:41:12 AM

*/

#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<ll> vi;
typedef map<ll, ll> mllmp;

#define faster                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);
#define make_pair mp

const ll mod = 1e9 + 7;

#define rep(i, j, k) for (ll i = j; i < k; i++)
#define repv(i, j, k) for (ll i = j; i > k; i--)

int main() {
  faster;
  ll x, y;
  vector<vector<int>> v(5, vector<int>(5));
  rep(i, 0, 5) rep(j, 0, 5) {
    cin >> v[i][j];
    if (v[i][j] == 1) {
      x = i;
      y = j;
    }
  }
  if (x == 2 && y == 2)
    cout << "0";
  else if (x == 2)
    cout << abs(y - 2);
  else if (y == 2)
    cout << abs(x - 2);
  else
    cout << abs(y - 2) + abs(x - 2);
  return 0;
}