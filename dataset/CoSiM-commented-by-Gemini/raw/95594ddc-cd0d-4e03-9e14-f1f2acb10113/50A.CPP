/**
 * @file 50A.CPP
 * @brief Calculates the ceiling of the division of two numbers.
 * @warning The filename `50A.CPP` is misleading. This code does not solve the
 *          "Domino piling" problem (Codeforces 50A). Instead, it implements
 *          the ceiling function `ceil(k / n)`.
 *
 * @details
 * This program reads two integers, `k` and `n`, and calculates the smallest integer
 * greater than or equal to the floating-point division of `k` by `n`. This is
 * a common pattern for solving problems that require distributing `k` items into
 * groups of size `n`.
 *
 * @algorithm Ceiling Division
 * @complexity
 * - Time: O(1).
 * - Space: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

// A collection of macros and typedefs common in competitive programming templates.
typedef long long ll;
#define ten5 100000 + 10
#define MOD 1000000007
#define rep(i, a, n) for (int i = a; i < n; i++)
#define iif(c, t, f) ((c) ? (t) : (f))
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
#define mp make_pair

// Unused comparison function.
int intcmp(const void *v1, const void *v2) { return *(int *)v1 - *(int *)v2; }

ll n, k;

int main() {
  cin >> n >> k;
  
  /**
   * @brief Functional Utility: Calculates ceil(k / n) using integer arithmetic.
   * Logic:
   * - `k / n`: Standard integer division gives the floor of the result.
   * - `k % n`: Finds the remainder.
   * - `!!(k % n)`: This expression evaluates to 1 if there is a non-zero
   *   remainder, and 0 otherwise.
   * By adding these two parts, we effectively compute the ceiling of the division.
   */
  cout << (k / n) + !!(k % n) << endl;

  return 0;
}