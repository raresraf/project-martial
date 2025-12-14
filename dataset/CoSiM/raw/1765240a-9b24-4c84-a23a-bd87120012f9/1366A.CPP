/*
   Created by Rahul Goel.
*/
/*******************************************************************************/
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
/*******************************************************************************/
using namespace std;
typedef long long ll;
typedef long double ld;
ll mod = 1000000007;
const int INF = 1e9;
const ll LINF = 1e18;
/*******************************************************************************/
/* For PBDS */
/*Two new methods X.order_of_key() and X.find_by_order()*/
/*
#include <bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag,
tree_order_statistics_node_update>;
*/

/*******************************************************************************/
#ifdef ONLINE_JUDGE
#define endl '\n'
#endif
#define fastio                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0)
#define all(c) (c).begin(), (c).end()
#define pii pair<ll, ll>
#define vi vector<ll>
#define vb vector<bool>
#define vvi vector<vector<ll>>
#define vvb vector<vector<bool>>
#define vpii vector<pair<ll, ll>>
#define vvpii vector<vector<pair<ll, ll>>>
#define present(c, x) ((c).find(x) != (c).end())
#define cpresent(c, x) (find(all(c), x) != (c).end())
#define uniq(c) (c).resize(distance((c).begin(), unique(all(c))))
#define min_pq(t) priority_queue<t, vector<t>, greater<t>>
#define max_pq(t) priority_queue<t>
#define pb push_back
#define ff first
#define ss second
/*******************************************************************************/
ll mod_sum() { return 0LL; }
template <typename T, typename... Args> T mod_sum(T a, Args... args) {
  return ((a + mod_sum(args...)) % mod + mod) % mod;
}

ll mod_prod() { return 1LL; }
template <typename T, typename... Args> T mod_prod(T a, Args... args) {
  return (a * mod_prod(args...)) % mod;
}
/*******************************************************************************/

/*
    Code begins after this.
*/

ll solve() {
  ll a, b;
  cin >> a >> b;
  if (a > b)
    swap(a, b);
  if (2 * a <= b)
    cout << a << endl;
  else {
    ll x = (2 * b - a) / 3;
    cout << x + min(b - 2 * x, (a - x) / 2) << endl;
  }
  return 0;
}

int main() {
  fastio;

  ll t = 1;
  cin >> t;
  while (t--)
    solve();

  return 0;
}
