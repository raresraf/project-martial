/**
 * @file 1426C.CPP
 * @brief Solution to Codeforces problem 1426C - Increase and Copy.
 * @description This program finds the minimum number of moves to obtain a sum of at least 'n'
 *              starting from the number 1. The allowed moves are incrementing a number by 1,
 *              or creating a copy of an existing number.
 *
 * Algorithm: The strategy is to find an optimal balance between "increase" moves and "copy" moves.
 * Let's say we decide to make `c` copies. This results in `c+1` identical numbers in total.
 * To reach a total value of at least `n`, each of these `c+1` numbers must have a value of at
 * least `ceil(n / (c+1))`.
 *
 * The number of moves to achieve this state is:
 * 1. `v = ceil(n / (c+1)) - 1` increase moves to get the initial '1' up to the target value.
 * 2. `c` copy moves.
 * Total moves = `c + v` = `c + ceil(n / (c+1)) - 1`.
 *
 * The program iterates through possible numbers of copies `c` (from 0 upwards) and calculates
 * the total moves for each case, finding the minimum. The loop variable `i` in the code
 * represents `c+1`. The loop runs up to 100,000, which is sufficient because the minimum
 * of `(i-1) + n/i - 1` occurs around `i = sqrt(n)`, and for n=10^9, sqrt(n) is ~31622.
 *
 * Time Complexity: O(sqrt(N_max)) for each test case, where N_max is the constraint on n.
 * Space Complexity: O(1).
 */

/*
    Created by Rahul Goel.
*/
// --- GENERIC COMPETITIVE PROGRAMMING TEMPLATE ---
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
/*******************************************************************************/
using namespace std;
using namespace __gnu_pbds;
template <typename T>
using ordered_set =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
/*******************************************************************************/
using ll = long long;
using ld = long double;
const ll MOD = 1000000007;
// const ll MOD = 998244353;
const int INF = 1e9;
const ll LINF = 1e18;
/*******************************************************************************/
ll mod_sum() { return 0LL; }
template <typename T, typename... Args> T mod_sum(T a, Args... args) {
  return ((a + mod_sum(args...)) % MOD + MOD) % MOD;
}
/*******************************************************************************/
ll mod_prod() { return 1LL; }
template <typename T, typename... Args> T mod_prod(T a, Args... args) {
  return (a * mod_prod(args...)) % MOD;
}
/*******************************************************************************/
#ifdef ONLINE_JUDGE
#define endl '
'
#endif
#define fastio                                                                 
  ios_base::sync_with_stdio(0);                                                
  cin.tie(0);                                                                  
  cout.tie(0)
#define all(c) (c).begin(), (c).end()
#define present(c, x) ((c).find(x) != (c).end())
#define cpresent(c, x) (find(all(c), x) != (c).end())
#define uniq(c) (c).resize(distance((c).begin(), unique(all(c))))
#define min_pq(t) priority_queue<t, vector<t>, greater<t>>
#define max_pq(t) priority_queue<t>
#define pb push_back
#define ff first
#define ss second
/*******************************************************************************/
using pii = pair<ll, ll>;
using vi = vector<ll>;
using vb = vector<bool>;
using vvi = vector<vector<ll>>;
using vvb = vector<vector<bool>>;
using vpii = vector<pii>;
using vvpii = vector<vector<pii>>;
/*******************************************************************************/
//.-.. . -. -.- .- .. ... .-.. --- ...- .
/*
    Code begins after this.
*/

ll solve() {
  ll n;
  cin >> n;

  ll ans = LINF;
  /**
   * Block Logic: Iterate through `i`, the number of items we aim to create.
   * This implies we perform `i-1` copy moves. The loop is bounded at 100,000,
   * which is sufficient to find the minimum for the given constraints of n.
   */
  for (ll i = 1; i <= 100000; i++) {
    // To reach a sum of `n` with `i` items, each item's value `x` must
    // be at least `ceil(n/i)`.
    ll x = (n + i - 1) / i;

    // Total moves = (number of 'increase' moves) + (number of 'copy' moves)
    // Increase moves = x - 1 (to get the first item from 1 to x)
    // Copy moves = i - 1 (to create i-1 additional items)
    // The expression x + (i - 1) - 1 calculates this total.
    ans = min(ans, x + (i - 1) - 1);
  }

  cout << ans << endl;

  return 0;
}

signed main() {
  fastio;

  ll t = 1;
  cin >> t;
  while (t--) {
    solve();
  }
  return 0;
}
