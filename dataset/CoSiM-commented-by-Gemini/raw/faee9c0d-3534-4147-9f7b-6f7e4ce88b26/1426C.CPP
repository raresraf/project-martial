/**
 * @file 1426C.CPP
 * @brief Solves the "Increase and Copy" problem (Codeforces 1426C).
 *
 * @description
 * The problem asks for the minimum number of moves to obtain at least 'n' items,
 * starting with one item. The available moves are:
 * 1. Increment the number of items by 1.
 * 2. Copy all current items, effectively doubling the count. However, the logic here
 *    interprets it as creating 'x' items and then making copies.
 *
 * @algorithm
 * The core idea is to find an optimal number of increment operations before starting
 * to make copies. Let's say we increment `x-1` times to get a value of `x`. This takes
 * `x-1` moves. Then, to reach at least `n` items, we need to copy `ceil(n/x) - 1`
 * times. In integer arithmetic, `ceil(n/x)` can be calculated as `(n-1)/x + 1`.
 * The total number of moves is the sum of increments and copies: `(x-1) + ((n-1)/x + 1) - 1 = (x-1) + (n-1)/x`.
 * This expression is minimized when `x` is close to `sqrt(n)`. This solution
 * calculates the number of moves by setting `x = floor(sqrt(n))`.
 *
 * @time_complexity O(T), where T is the number of test cases. Each test case is solved in O(1).
 * @space_complexity O(1).
 */

/// In the name of Allah SWT

using namespace std;
#include <bits/stdc++.h>

//==================================================================================
// Section: Macros and Type-defs for Competitive Programming
//----------------------------------------------------------------------------------
// These macros are common in competitive programming to reduce boilerplate code
// and write solutions faster. They include shorthands for data types, I/O
// operations, loops, and mathematical constants.
//==================================================================================

#define ll long long int
#define dd double

#define scl(x) scanf("%lld", &x)
#define scll(x, y) scanf("%lld %lld", &x, &y)
#define scd(x) scanf("%lf", &x)
#define scdd(x, y) scanf("%lf %lf", &x, &y)

#define prl(x) printf("%lld
", x)
#define prll(x, y) printf("%lld %lld
", x, y)
#define prys printf("YES
")
#define prno printf("NO
")

#define ON(n, i) (n | (1LL << i))
#define OFF(n, i) (n & (~(1LL << i)))
#define CHK(n, i) (n & (1LL << i))

#define For(i, x, y) for (ll i = x; i < y; i++)
#define Mem(ara, x) memset(ara, x, sizeof(ara))

#define pb push_back
#define pll pair<ll, ll>
#define ff first
#define ss second

#define maxn 200005 /// 2x10^5 + 5
//#define maxn 1000006 ///10^6 + 6
//#define maxn 1000000009 ///10^9 + 9

#define pi acos(-1.00)
#define eps 0.0000000001 /// 10^-10
#define inf LONG_LONG_MAX
#define mod 1000000007 /// 10^9+7

//==================================================================================
// Section: Global Variables
//----------------------------------------------------------------------------------
// Global variables used across different functions for test cases and problem inputs.
//==================================================================================

ll t, test, temp;
ll n, m, k, kount;
ll a, b, c, ans, u, v;
ll x, y, z = -1, maxi, mini;

/**
 * @brief Solves a single test case for the "Increase and Copy" problem.
 */
void solve() {
  scl(n);

  // Find the optimal number of increments to perform before copying.
  // This is the integer part of sqrt(n).
  x = sqrt(n);

  // Calculate the minimum moves.
  // (x-1) moves to increment up to x.
  // (n-1)/x moves to copy x until we have at least n items.
  ans = x - 1;
  ans += (n - 1) / x;

  prl(ans);
}

/**
 * @brief Main function to handle test cases.
 */
int main() {
  test = 1;
  // This loop structure is a common competitive programming pattern
  // to handle multiple test cases read from standard input.
  while (scl(test) == 1)
    while (test--)
      solve();
}
