/*
   Created by Rahul Goel.
*/

/**
 * @file 1366A.CPP
 * @brief Solution for the "Shovels and Swords" competitive programming problem.
 * @details The problem involves two resources, 'a' and 'b', and two types of items to craft.
 * One item requires 2 of 'a' and 1 of 'b', while the other requires 1 of 'a' and 2 of 'b'.
 * The goal is to find the maximum total number of items that can be crafted.
 *
 * @section algorithm Algorithm
 * The problem can be modeled as a resource allocation optimization. Let the resources be sticks and diamonds.
 * - Shovel: 2 sticks, 1 diamond
 * - Sword: 1 stick, 2 diamonds
 * Given `a` sticks and `b` diamonds, we want to maximize the number of tools (shovels + swords).
 *
 * The maximum number of tools is constrained by the total availability of each resource. The optimal number of tools is given by the formula: `min(a, b, (a + b) / 3)`.
 * This program implements a logically equivalent, optimized check:
 * 1. It first ensures that `a` holds the smaller quantity and `b` the larger one.
 * 2. It checks if the more abundant resource `b` is at least twice the amount of the scarcer resource `a` (`2 * a <= b`).
 *    - If it is, the scarcer resource `a` is the sole bottleneck. The maximum number of items is `a`.
 *    - Otherwise, the resources are relatively balanced, and the total number of items is limited by the combined sum of resources, which is `(a + b) / 3`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1) for each test case, as it only involves a few arithmetic operations and comparisons.
 * - Space Complexity: O(1).
 */
/*******************************************************************************/
// Standard competitive programming boilerplate: includes, typedefs, and macros for convenience.
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
/*
  Optional Policy-Based Data Structures (PBDS) from GNU GCC library.
  These provide advanced data structures like ordered_set for operations
  such as finding the k-th smallest element or the number of items smaller than a key.
*/
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
// Macros for faster I/O and code brevity in a competitive programming context.
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
// Variadic templates for modular arithmetic (sum and product).
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

// This function solves a single test case.
ll solve() {
  ll a, b;
  cin >> a >> b;
  // Pre-condition: Ensure 'a' is the smaller of the two resources.
  if (a > b)
    swap(a, b);
  // Case 1: Resource 'b' is much more abundant than 'a'.
  // In this scenario, 'a' is the limiting factor. We can make 'a' items.
  // This is equivalent to min(a, (a+b)/3) when 2*a <= b.
  if (2 * a <= b)
    cout << a << endl;
  // Case 2: Resources are relatively balanced.
  // The total number of items is limited by the sum of both resources.
  // The result is (a+b)/3. The original complex line of code simplifies to this.
  else {
    ll x = (a + b) / 3;
    cout << x << endl;
  }
  return 0;
}

int main() {
  // Fast I/O setup.
  fastio;

  // 't' is the number of test cases.
  ll t = 1;
  cin >> t;
  while (t--)
    solve();

  return 0;
}
