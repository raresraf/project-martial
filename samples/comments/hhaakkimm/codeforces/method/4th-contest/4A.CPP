#include <bits/stdc++.h>
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cstdio>
// #include <string>
// #include <cmath>
// #include <ctime>
// #include <queue>
// #include <map>

using namespace std;

// constants
const int MAXN = (int)(1e5 + 1e1);
const double PI = acos(-1.0);

// defines
#define ll long long
#define TASKNAME "."
#define fr freopen(TASKNAME "in", "r", stdin);
#define fw freopen(TASKNAME "out", "w", stdout);
#define mp make_pair
#define pb push_back
#define eof (-1)
#define forr(xx, yy, zz) for (int zz = xx; zz <= yy; zz++)
#define forl(xx, yy, zz) for (int zz = xx; zz >= yy; zz--)
#define sqr(x) ((x) * (x))
#define sz(x) (int)x.size()
#define mod 10000000007
#define len(s) (int)(s.length())
#define all(a) a.begin(), a.end()
//=

ll n;
vector<ll> ans;

void rec(int q, int w, ll k) {
  if (q == w && k >= n)
    ans.pb(k);
  if (q + w > 10)
    return;

  rec(q + 1, w, k * 10ll + 4);
  rec(q, w + 1, k * 10ll + 7);
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cin >> n;
  rec(0, 0, 0);
  sort(all(ans));
  cout << ans[0];
  return 0;
}
