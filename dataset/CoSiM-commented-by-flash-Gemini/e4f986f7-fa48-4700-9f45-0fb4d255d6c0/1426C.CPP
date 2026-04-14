/**
 * @file 1426C.CPP
 * @brief This program solves the "Increase and Copy" problem from competitive programming.
 *        The goal is to find the minimum number of operations to obtain at least 'n' copies of an item.
 *        Starting with one item, available operations are:
 *        1. Increment the value of the current item by 1.
 *        2. Copy the current item, which doubles the number of items.
 *
 * Algorithm:
 * The problem can be rephrased as finding an optimal intermediate value 'i' to reach by incrementing,
 * and then performing copy operations to get at least 'n' total items.
 * If we increment the initial item (value 1) to 'i', this takes `i - 1` operations.
 * Once we have 'i' items, we need to perform `x` copy operations such that `i * (x + 1) >= n`.
 * The minimum number of copies needed for a base value `i` is `ceil(n / i)`. Let this be `x_needed`.
 * The total operations would then be `(i - 1)` (to get value `i`) + `(x_needed - 1)` (copy operations).
 * This simplifies to `i - 1 + (ceil(n/i) - 1) = i + ceil(n/i) - 2`.
 * The algorithm iterates through possible values of 'i' from 1 up to a certain limit (heuristically `sqrt(n)` or a larger constant like `10^5` to cover edge cases or large 'n'),
 * calculating the total operations for each 'i' and finding the minimum.
 *
 * Time Complexity: O(sqrt(N)) in theory, but in this implementation, it's O(LIMIT) where LIMIT is 100000.
 * Space Complexity: O(1) as only a few variables are used.
 */
/*
    Created by Rahul Goel.
*/
#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
// For ordered set/tree data structures provided by GNU extension.
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
/*******************************************************************************/
using namespace std;         // Uses the standard namespace to avoid prefixing std::.
using namespace __gnu_pbds;  // Uses GNU's policy-based data structures namespace.
template <typename T>
using ordered_set =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
/*******************************************************************************/
using ll = long long;   // Type alias for long long integers.
using ld = long double; // Type alias for long double floating-point numbers.
const ll MOD = 1000000007; // A common modulo constant, used in problems involving large numbers.
// const ll MOD = 998244353; // Another common modulo constant.
const int INF = 1e9;    // Represents infinity for integer values.
const ll LINF = 1e18;   // Represents infinity for long long values.
/*******************************************************************************/
// Helper function for modular sum (variadic template).
ll mod_sum() { return 0LL; }
template <typename T, typename... Args> T mod_sum(T a, Args... args) {
  return ((a + mod_sum(args...)) % MOD + MOD) % MOD;
}
/*******************************************************************************/
// Helper function for modular product (variadic template).
ll mod_prod() { return 1LL; }
template <typename T, typename... Args> T mod_prod(T a, Args... args) {
  return (a * mod_prod(args...)) % MOD;
}
/*******************************************************************************/
#ifdef ONLINE_JUDGE
#define endl '\n' // Use '\n' instead of std::endl for faster output in online judges.
#endif
#define fastio                                                                 \
  ios_base::sync_with_stdio(0); /* Untie C-style I/O from C++ streams */       \
  cin.tie(0);                   /* Untie cin from cout */                      \
  cout.tie(0)                   /* Untie cout from cin */
#define all(c) (c).begin(), (c).end()                       // Shorthand for iterators to the beginning and end of a container.
#define present(c, x) ((c).find(x) != (c).end())            // Checks if an element 'x' is present in an associative container 'c'.
#define cpresent(c, x) (find(all(c), x) != (c).end())       // Checks if an element 'x' is present in a sequential container 'c'.
#define uniq(c) (c).resize(distance((c).begin(), unique(all(c)))) // Removes duplicate elements from a sorted container.
#define min_pq(t) priority_queue<t, vector<t>, greater<t>> // Min-priority queue.
#define max_pq(t) priority_queue<t>                         // Max-priority queue.
#define pb push_back                                        // Shorthand for push_back.
#define ff first                                            // Shorthand for pair.first.
#define ss second                                           // Shorthand for pair.second.
/*******************************************************************************/
// Type aliases for common data structures using long long.
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

/**
 * @brief Solves a single test case of the "Increase and Copy" problem.
 * Functional Utility: Calculates the minimum operations to achieve at least 'n' copies.
 * @return 0 (though the problem usually expects printing the answer, not returning).
 */
ll solve() {
  ll n;
  // Reads the target number of copies 'n'.
  cin >> n;

  ll ans = LINF; // Initializes the minimum operations with a very large value.

  /**
   * Block Logic: Iterates through possible values 'i' that can be reached by incrementing.
   * 'i' represents the value we achieve by `i-1` increment operations.
   * For each 'i', it calculates the number of copy operations needed and updates the minimum answer.
   * The loop limit `100000` is an empirical bound that typically covers the optimal `i` for typical competitive programming constraints of 'n'.
   * Invariant: `ans` stores the minimum operations found so far.
   */
  for (ll i = 1; i <= 100000; i++) {
    // Calculates 'x', the ceiling of n/i. This is the number of copies (multiples of i) needed.
    // Equivalent to ceil(n/i).
    ll x = (n + i - 1) / i;
    // Updates 'ans' with the minimum of current 'ans' and the total operations for this 'i'.
    // Total operations = (i - 1) increments + (x - 1) copy operations.
    ans = min(ans, x + (i - 1) - 1); // Simplified to ans = min(ans, x + i - 2)
  }

  // Prints the minimum number of operations for the current test case.
  cout << ans << endl;

  return 0;
}

/**
 * @brief Main function: Sets up fast I/O and handles multiple test cases.
 * Functional Utility: Entry point of the program, reads number of test cases and calls solve() for each.
 */
signed main() {
  fastio; // Enables fast I/O.

  ll t = 1;
  // Reads the number of test cases.
  cin >> t;
  // Block Logic: Loops through each test case and calls the solve function.
  while (t--) {
    solve();
  }
  return 0; // Standard program exit.
}
