/**
 * @file 1366A.cpp
 * @brief Solution for a resource allocation problem, likely from a competitive programming context.
 * @details The problem involves two types of resources, say `a` and `b`. We need to create two types of items.
 * Type 1 requires 1 unit of `a` and 2 units of `b`.
 * Type 2 requires 2 units of `a` and 1 unit of `b`.
 * The goal is to find the maximum total number of items that can be created.
 *
 * @section algorithm Algorithm
 * The solution is based on a greedy approach and mathematical observation.
 * 1. Let `a` and `b` be the quantities of the two resources.
 * 2. Ensure `a` is the smaller quantity by swapping if necessary. This simplifies the logic.
 * 3. The most efficient way to use the resources is to create an equal number of Type 1 and Type 2 items.
 *    Each pair of (Type 1, Type 2) items requires 3 units of `a` and 3 units of `b`.
 *    So, we can make `k = min(a, b) / 3` pairs.
 * 4. A more direct approach is to see that to make one item of either type, we need at least 1 `a` and 1 `b`.
 *    To make `k` items, we need at least `k` of `a` and `k` of `b`. The total resources needed are `3k`.
 *    So, `k` cannot be more than `(a+b)/3`.
 *    Also, `k` cannot be more than `min(a,b)`.
 *    So, the maximum number of items is `min(a, b, (a+b)/3)`.
 *    Since `(a+b)/3` can be smaller than `min(a,b)` (e.g., a=1, b=100), this is a key constraint.
 *    The logic in the code is a bit different but arrives at the same result.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1) for each test case, as it involves a few arithmetic operations and comparisons.
 * - Space Complexity: O(1), as it uses a fixed amount of memory.
 */
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

/**
 * @brief Solves a single test case.
 * @return 0 on completion.
 */
ll solve() {
  ll a, b;
  cin >> a >> b;
  // The constraints are symmetric, so we can assume a <= b to simplify.
  if (a > b)
    swap(a, b);
  // The maximum number of items we can make is limited by the minimum of the two resources,
  // and also by the total amount of resources (since each item costs 3 total).
  // The answer is min(a, b, (a+b)/3).
  // The code implements this logic in a slightly different way.
  cout << min({a, b, (a + b) / 3}) << endl;

  return 0;
}

/**
 * @brief Main function to handle multiple test cases.
 * @return 0 on successful execution.
 */
int main() {
  fastio;

  ll t = 1;
  cin >> t;
  while (t--)
    solve();

  return 0;
}

