/**
 * @file 1426C.CPP
 * @brief Solution to Codeforces Problem 1426C: "Minimizing the maximum number of blocks".
 *
 * Algorithm: The problem asks for the minimum number of moves to reach a target number 'n' starting from '1'.
 * A move can either be incrementing the current number by 1, or multiplying the current number by some integer 'k'.
 * This solution uses binary search to find the optimal number of initial increments ('a' in `tryit`, or `mid` in `func`)
 * that minimizes the total operations. If we perform `x-1` initial increment operations, we reach `x`.
 * Then, to reach `n` from `x`, we need `ceil(n/x) - 1` additional operations (either adding `x` repeatedly or multiplying).
 * The total moves for a chosen `x` is `(x - 1) + ceil((n - 1) / x)`.
 * The binary search effectively finds the `x` (or `mid+1`) that minimizes this expression.
 * The range for binary search for `mid` can be up to `sqrt(N)` or slightly more for optimality,
 * but a large enough upper bound (e.g., 1e10) is safe since the function `(x-1) + (N-1)/x` is convex.
 *
 * Time Complexity: O(TC * log N), where TC is the number of test cases and N is the input number.
 * Space Complexity: O(1).
 * Domain: Competitive Programming, Binary Search, Optimization, Number Theory.
 * Problem Link: https://codeforces.com/contest/1426/problem/C
 */
#include <bits/stdc++.h> // Functional Utility: Includes almost all standard C++ library headers.

using namespace std; // Functional Utility: Brings all names from the std namespace into the current scope.

// Functional Utility: Defines long long as 'll' for brevity.
#define ll long long int
// Functional Utility: Shorthand for std::make_pair.
#define mp make_pair
// Functional Utility: Shorthand for std::push_back.
#define pb push_back
// Functional Utility: Shorthand for std::pair::first.
#define fi first
// Functional Utility: Shorthand for std::pair::second.
#define si second
/**
 * @brief Optimizes C++ standard input/output streams by decoupling them from C standard streams
 * and disabling synchronization. This significantly speeds up I/O operations, crucial for
 * competitive programming performance.
 */
#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
/**
 * @brief Macro for reading 'n' elements into a 0-indexed array 'a'.
 * Functional Utility: Simplifies input loops for arrays.
 * @param a Array to read into.
 * @param n Number of elements.
 */
#define F(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cin >> a[i];                                                               \
  }
/**
 * @brief Macro for reading 'n' elements into a 1-indexed array 'a'.
 * Functional Utility: Simplifies input loops for arrays.
 * @param a Array to read into.
 * @param n Number of elements.
 */
#define F1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cin >> a[i];                                                               \
  }
/**
 * @brief Macro for printing 'n' elements of a 0-indexed array 'a'.
 * Functional Utility: Simplifies output loops for arrays.
 * @param a Array to print.
 * @param n Number of elements.
 */
#define P(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for printing 'n' elements of a 1-indexed array 'a'.
 * Functional Utility: Simplifies output loops for arrays.
 * @param a Array to print.
 * @param n Number of elements.
 */
#define P1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for reading elements into a 0-indexed 2D array 'a' of size n x m.
 * Functional Utility: Simplifies input loops for 2D arrays.
 * @param a 2D array to read into.
 * @param n Number of rows.
 * @param m Number of columns.
 */
#define NF(a, n, m)                                                            \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
/**
 * @brief Macro for reading elements into a 1-indexed 2D array 'a' of size n x m.
 * Functional Utility: Simplifies input loops for 2D arrays.
 * @param a 2D array to read into.
 * @param n Number of rows.
 * @param m Number of columns.
 */
#define NF1(a, n, m)                                                           \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
/**
 * @brief Macro for printing elements of a 0-indexed 2D array 'a' of size n x m.
 * Functional Utility: Simplifies output loops for 2D arrays.
 * @param a 2D array to print.
 * @param n Number of rows.
 * @param m Number of columns.
 */
#define PNF(a, n, m)                                                           \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for printing elements of a 1-indexed 2D array 'a' of size n x m.
 * Functional Utility: Simplifies output loops for 2D arrays.
 * @param a 2D array to print.
 * @param n Number of rows.
 * @param m Number of columns.
 */
#define PNF1(a, n, m)                                                          \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for integer ceiling division.
 * Algorithm: Calculates `ceil(x/y)` using integer arithmetic.
 * @param x Numerator.
 * @param y Denominator.
 * @return `ceil(x/y)`.
 */
#define ceil_div(x, y) (((x) + (y)-1) / (y))
// Constant for a maximum value, likely array size or problem constraint.
const int nax = 1e7;
// Constant for a modulo value, common in problems involving large numbers.
const int mod = 1e9 + 7;
// Global variable to store the target number 'n' for the problem.
ll n;

/**
 * @brief Calculates the cost (number of moves) if 'a' is the current number
 *        and we need to reach 'n'. This function is defined but not used in the main logic.
 *        It seems to model a specific path: if 'a' is current, what's the cost to reach 'n'
 *        by incrementing 'a' until it becomes `n` (if `a >= n`), or by first incrementing to `a+1`
 *        and then multiplying by `a+1` until `n` is reached.
 * @param a A number representing a potential intermediate state or number of initial increments.
 * @return The calculated cost.
 */
ll tryit(ll a) {
  if (a >= n) {
    return a;
  }
  // This line implies: cost to reach `a+1` (which is `a` increments)
  // plus the number of multiplications of `a+1` needed to reach `n`.
  // `ceil_div(n, a + 1) - 1` represents the number of additional "multiply by (a+1)" operations.
  ll b = ceil_div(n, a + 1) - 1;
  return a + b;
}

/**
 * @brief Solves the "Beautiful Matrix" problem (Codeforces 1426C) using binary search.
 * It finds the minimum number of operations to reach 'n' from '1'.
 * The binary search is performed over `mid` where `mid` represents the number of initial
 * "add 1" operations, making the current number `mid + 1`. The goal is to minimize
 * `(mid) + ceil((n - 1) / (mid + 1))`.
 */
void func() {
  cin >> n; // Read the target number 'n'.
  // Functional Utility: Defines the binary search range [s, e]. 'e' is a large upper bound.
  // 'ans' stores the minimum operations found.
  ll s = 0, e = 1e10, ans;
  // Block Logic: Standard binary search loop to find the optimal 'mid'.
  // Invariant: `ans` always holds the minimum number of moves found so far.
  while (s <= e) {
    ll mid = (s + e) / 2;
    // Functional Utility: `(1 + mid) * (1 + mid)` approximates the square of the current value
    // after `mid` increments. This helps in finding the `mid` where `mid+1` is roughly `sqrt(n)`.
    ll val = (1 + mid) * (1 + mid);
    // Block Logic: If `(mid+1)^2` is greater than or equal to `n`, then `mid+1` is a valid
    // candidate for the intermediate number `x`.
    if (val >= n) {
      // Functional Utility: Calculates the total cost for this `mid`.
      // `mid` is the number of 'add 1' operations.
      // `ceil_div(n, mid + 1) - 1` is the number of 'multiply by (mid+1)' operations.
      ans = mid + ceil_div(n, mid + 1) - 1;
      // Block Logic: Try to find a smaller `mid` that also satisfies the condition.
      e = mid - 1;
    } else {
      // Block Logic: If `(mid+1)^2` is too small, need more increments, so search in upper half.
      s = mid + 1;
    }
  }
  cout << ans << endl; // Output the minimum number of moves.
}

int main() {
  fastIO // Functional Utility: Applies I/O optimization.
// Block Logic: Conditional compilation for local file I/O (not used in online judge).
#ifndef ONLINE_JUDGE
      freopen("../inp.txt", "r", stdin); // Redirects stdin to "inp.txt".
  freopen("../out.txt", "w", stdout);    // Redirects stdout to "out.txt".
#endif
  int t;
  cin >> t; // Read the number of test cases.
  // Block Logic: Loop through each test case.
  while (t--) {
    func(); // Call the function to solve the problem for the current test case.
  }
}