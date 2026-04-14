/**
 * @file 1426C.CPP
 * @brief This program determines the minimum number of operations (`cnt`)
 *        required to make an accumulating `sum` reach or exceed a target value `n`.
 *        The process starts with `sum = 1` and a `step_value = 1`.
 *
 * Operations are defined as:
 * 1. Increment `sum` by the current `step_value`.
 * 2. If `sum` happens to equal the product of `step_value * (step_value + 1)`
 *    at the beginning of an iteration, then `step_value` is first incremented
 *    by one, and then `sum` is increased by this new `step_value`.
 * Each operation increases the `cnt` by one.
 *
 * Algorithm:
 * Iterative simulation of the accumulation process. The loop continues as long
 * as `sum` is less than `n`. Inside the loop, it first checks a specific condition
 * involving `sum` and `step_value` to potentially increment `step_value`.
 * Then, `sum` is increased by the (possibly new) `step_value`, and the operation count `cnt`
 * is incremented.
 *
 * Time Complexity: The `sum` grows quadratically if `i` increments frequently (e.g., `1, 2, 4, 7, 11, ...`).
 * In the worst case, if `i` never increments, `sum` increases linearly, making it O(N).
 * In general, `sum` roughly follows `i*(i+1)*(i+2)/3`, so `i` grows as `N^(1/3)`.
 * The number of operations `cnt` is roughly proportional to `i`. So, Time Complexity is approximately O(N^(1/3)).
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
#include <iostream>      // Required for standard input/output functions.
using namespace std;     // Uses the standard namespace to avoid prefixing std::.

// Macros for competitive programming convenience.
#define rep(i, a, b) for (int i = a; i < b; i++)       // Loop from a (inclusive) to b (exclusive).
#define repe(i, a, b) for (int i = a; i <= b; i++)     // Loop from a (inclusive) to b (inclusive).
#define ll long long                                   // Type alias for long long integers.
#define ui unsigned int                                // Type alias for unsigned int.
#define pb push_back                                   // Shorthand for std::vector::push_back.
#define deb(x) cout << #x << '=' << x << endl          // Debug macro to print variable name and value.
#define deb2(x, y) cout << #x << '=' << x << #y << '=' << y << endl // Debug macro for two variables.
#define all(x) x.begin(), x.end()                      // Shorthand for iterators to the beginning and end of a container.
#define endl "\n"                                      // Faster newline output.
#define ff first                                       // Shorthand for std::pair::first.
#define ss second                                      // Shorthand for std::pair::second.
#define lb lower_bound                                 // Shorthand for std::lower_bound.
#define ub upper_bound                                 // Shorthand for std::upper_bound.
#define INF (int)1e9                                   // Represents integer infinity.
#define PI 3.1415926535897932384626433832795            // Value of PI.
#define MOD 1000000007                                 // A common modulo constant.

// Built-in GCC extensions for bit manipulation.
#define hcf(x, y) __gcd(x, y)                        // Highest Common Factor (GCD).
#define lcm(x, y) (x * y) / (__gcd(x, y))            // Least Common Multiple.
#define CountOne(x) __builtin_popcount(x)            // Counts set bits (1s) in an integer.
#define parity(x)                                    \
  __builtin_parity(x) // Returns true if the number of set bits is odd.
#define LeadZero(x)                                  \
  __builtin_clz(x) // Counts leading zeros in binary representation.
#define TrailZero(x) __builtin_ctz(x) // Counts trailing zeros in binary representation.

// Macro for fast I/O.
#define fastIO                                                                 \
  ios::sync_with_stdio(0); /* Untie C-style I/O from C++ streams */           \
  cin.tie(0);              /* Untie cin from cout */                          \
  cout.tie(0)              /* Untie cout from cin */

// Type aliases for common data structures.
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<ll> vl;
typedef vector<vl> vvl;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<string, string> pss;
typedef map<int, int> mii;
typedef unordered_map<int, int> umap_ii;
typedef unordered_map<int, int> umap_ii; // Duplicate definition.
typedef unordered_map<string, int> umap_si;
mt19937_64
    rang(chrono::high_resolution_clock::now().time_since_epoch().count()); // Random number generator engine.
// alt + ctrl + N

//===============================================================================//
/**
 * @brief Calculates x raised to the power y.
 * @param x The base.
 * @param y The exponent.
 * @return x^y.
 */
ll power(ll x, ll y) {
  ll v = 1;
  while (y > 0) {
    if (y & 1) // If y is odd.
      v = v * x;
    y = y >> 1; // y = y / 2.
    x = x * x;  // x = x^2.
  }
  return v;
}

/**
 * @brief Checks if a number is a power of two.
 * @param x The number to check.
 * @return True if x is a power of two, false otherwise.
 */
bool isPowerOfTwo(ll x) {
  return x && (!(x & (x - 1))); // Efficient check using bitwise operations.
}

//===============================================================================//

/**
 * @brief Main function: Sets up fast I/O, handles file redirection for local testing,
 *        and processes multiple test cases for the accumulation problem.
 * Functional Utility: Entry point of the program.
 */
int main() {
  fastIO; // Enables fast I/O.
  srand(chrono::high_resolution_clock::now().time_since_epoch().count()); // Seeds random number generator.

#ifndef ONLINE_JUDGE
  // Block Logic: Redirects standard input and output to files for local debugging.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  ll test = 1;
  // Reads the number of test cases.
  cin >> test;
  // Block Logic: Iterates through each test case.
  // Invariant: 'test' represents the remaining test cases.
  while (test--) {
    ll n; // Target sum value.
    // Reads the target sum 'n' for the current test case.
    cin >> n;
    ll sum = 1; // Current accumulated sum, starts at 1.
    ll i = 1;   // Current step value, starts at 1.
    ll cnt = 0; // Operation count.

    /**
     * Block Logic: Accumulates 'sum' until it reaches or exceeds 'n',
     * incrementing 'cnt' for each operation. The 'step_value' ('i')
     * conditionally increases based on 'sum' reaching `i * (i+1)`.
     * Pre-condition: `n` is a positive integer.
     * Invariant: `sum` is always increasing, and `cnt` tracks the number of operations.
     */
    while (sum < n) {
      if (sum == i * (i + 1)) { // Specific condition to increment step_value.
        i++;                    // Increment step_value.
        sum += i;               // Add the new step_value to sum.
        cnt++;                  // Increment operation count.
      } else {
        sum += i; // Add the current step_value to sum.
        cnt++;    // Increment operation count.
      }
    }
    // Prints the minimum number of operations to reach 'n'.
    cout << cnt << endl;
  }
  return 0; // Standard program exit.
}