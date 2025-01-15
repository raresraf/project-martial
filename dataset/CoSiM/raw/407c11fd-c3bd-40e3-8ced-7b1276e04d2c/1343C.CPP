/********************************************************************************
 *                                                                               *
 *        Author     : Rezoan Shakil Prince * Batch      : 14 * Email      :
 *prince.cuet.77@gmail.com                                  * University :
 *Chittagong University of Engineering and Technology       *
 *                                                                               *
 ********************************************************************************/

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

void solve() {
  ll n;
  while (cin >> n) {
    ll a[n];
    for (int i = 0; i < n; i++)
      cin >> a[i];

    ll mx = -1, mn = INT_MIN;
    bool bg = false, sm = false;
    if (a[0] > 0) {
      sm = true;
      mx = a[0];
    } else {
      bg = true;
      mn = a[0];
    }

    ll sum = 0;
    for (ll i = 1; i < n; i++) {
      if (a[i] > 0 && sm == true)
        mx = max(a[i], mx);
      if (a[i] < 0 && bg == true)
        mn = max(a[i], mn);

      if (a[i] < 0 && sm == true) {
        sum += mx;
        mn = a[i];
        mx = -1;

        bg = true;
        sm = false;
      } else if (a[i] > 0 && bg == true) {
        sum += mn;
        mn = INT_MIN;
        mx = a[i];

        bg = false;
        sm = true;
      }
    }

    if (bg)
      sum += mn;
    else if (sm)
      sum += mx;

    cout << sum << "\n";
  }
}

int main() {
  ios_base ::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  // #ifndef prince77
  //         freopen("input.txt", "r", stdin);
  //         freopen("output.txt", "w", stdout);
  // #endif

  int tc;
  cin >> tc;
  while (tc--)
    solve();

  return 0;
}
