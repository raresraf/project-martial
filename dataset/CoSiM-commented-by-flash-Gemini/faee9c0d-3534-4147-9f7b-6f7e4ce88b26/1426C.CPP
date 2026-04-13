/**
 * @file 1426C.CPP
 * @brief This program solves a problem related to minimizing operations to reach
 *        a target number 'n' starting from 1. The allowed operations are increment
 *        by 1, and multiply by 'k' (where 'k' is the current number). The solution
 *        uses a mathematical observation involving the square root of 'n'.
 * Algorithm: Optimized mathematical approach (related to number theory/greedy strategy).
 * Time Complexity: O(1) for each test case, as it involves a fixed number of arithmetic
 *                  operations and a square root calculation.
 * Space Complexity: O(1) for storing input variables and the result.
 */
/// In the name of Allah SWT

using namespace std;
#include <bits/stdc++.h>

#define ll long long int
#define dd double

// Macros for fast input (long long)
#define scl(x) scanf("%lld", &x)
#define scll(x, y) scanf("%lld %lld", &x, &y)
// Macro for fast input (double)
#define scd(x) scanf("%lf", &x)
#define scdd(x, y) scanf("%lf %lf", &x, &y)

// Macros for fast output (long long)
#define prl(x) printf("%lld\n", x)
#define prll(x, y) printf("%lld %lld\n", x, y)
// Macros for printing "YES" or "NO"
#define prys printf("YES\n")
#define prno printf("NO\n")

// Bit manipulation macros (unused in this specific code)
#define ON(n, i) (n | (1LL << i))
#define OFF(n, i) (n & (~(1LL << i)))
#define CHK(n, i) (n & (1LL << i))

// Loop macro (unused in this specific code)
#define For(i, x, y) for (ll i = x; i < y; i++)
// Memory set macro (unused in this specific code)
#define Mem(ara, x) memset(ara, x, sizeof(ara))

// Vector and pair macros
#define pb push_back
#define pll pair<ll, ll>
#define ff first
#define ss second

// Max array size (unused)
#define maxn 200005 /// 2x10^5 + 5
//#define maxn 1000006 ///10^6 + 6
//#define maxn 1000000009 ///10^9 + 9

// Mathematical constants
#define pi acos(-1.00)
#define eps 0.0000000001 /// 10^-10
#define inf LONG_LONG_MAX
#define mod 1000000007 /// 10^9+7

// Global variables (many unused)
ll t, test, temp;
ll n, m, k, kount;
ll a, b, c, ans, u, v;
ll x, y, z = -1, maxi, mini;

/**
 * @brief Solves a single test case by calculating the minimum operations.
 * Functional Utility: Computes the minimum number of operations to reach `n`
 *                     using a specific mathematical formula derived from optimization.
 * Pre-condition: `n` is a positive integer.
 */
void solve() {
  scl(n); // Reads the target number 'n'.
  // Calculates the integer part of the square root of 'n'.
  x = sqrt(n);

  /**
   * Block Logic: Calculates the minimum operations `ans`.
   * Invariant: `ans` is computed based on `x` (floor(sqrt(n)))
   *            representing an optimal split point for operations.
   *            `x - 1` represents increments to reach `x`.
   *            `(n - 1) / x` represents multiplications by `x` (plus increments to bridge remainder).
   */
  ans = x - 1;       // Cost to reach `x` from 1 (x-1 increments).
  ans += (n - 1) / x; // Cost to reach `n` from `x` (approximately (n-1)/x multiplications by `x`).

  prl(ans); // Prints the calculated minimum operations.
}

int main() {
  test = 1; // Default number of test cases.
  // Reads the number of test cases (if provided by input), then processes each test case.
  while (scl(test) == 1) // Reads 'test' from input; loop continues as long as input is valid.
    while (test--)       // Loop for each test case.
      solve();         // Calls the solve function for the current test case.
}
