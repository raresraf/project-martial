/**
 * @file 1140D.CPP
 * @brief This program calculates the sum of `i * (i+1)` for `i` ranging from 2 up to `n-1`.
 *   This is typically a problem involving mathematical series or combinatorics in competitive programming.
 *   Assumed problem context: Codeforces Round #547 (Div. 3), Problem D - "Cipher Shifer" (or similar structure).
 * * Algorithm: Iterative summation. A loop accumulates the product `i * (i+1)` for each `i` in the specified range.
 * Time Complexity: O(N), where N is the input integer `n`, due to the linear iteration.
 * Space Complexity: O(1) - constant space for a few integer variables.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Macro to optimize C++ standard I/O operations.
 * Functional Utility: Disables synchronization between C-style I/O and C++ streams (`ios::sync_with_stdio(0)`)
 *                     and unties `cin` from `cout` (`cin.tie(0)`, `cout.tie(0)`),
 *                     significantly speeding up input/output operations for competitive programming.
 */
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n" // Defines 'endl' as a newline character for potentially faster output than std::endl (which also flushes).
#define forn(i, n) for (int i = 0; i < int(n); ++i) // Macro for a common for-loop pattern.
#define vet(a)                                                                 \
  for (auto &i : a)                                                            \
  cout << i << " " // Macro to print elements of a container.
#define out(a) cout << a << endl // Macro to print a value followed by a newline.

typedef long long int ll; // Typedef for long long integer, useful for larger sums.

/**
 * @brief Main function to compute the sum of `i * (i+1)`.
 * Functional Utility: Sets up I/O optimizations, handles optional file I/O for local testing,
 *                     reads the upper bound `n`, calculates the sum, and prints the result.
 */
int32_t main() {
  // Functional Utility: Applies I/O optimizations defined by the IOS macro.
  IOS;
// Block Logic: Conditional compilation for redirecting input/output to files.
// This is common in competitive programming for local testing without modifying the submission code.
#ifndef ONLINE_JUDGE
  // Redirects standard input to "input.txt".
  freopen("input.txt", "r", stdin);
  // Redirects standard output to "output.txt".
  freopen("output.txt", "w", stdout);
#endif

  int n;
  // Block Logic: Reads the integer 'n' which defines the upper limit for the summation.
  cin >> n;
  ll ans = 0; // Initializes a long long integer to store the cumulative sum to prevent overflow.
  /**
   * Block Logic: Iterates from `i = 2` to `n-1` (exclusive of `n`) to calculate the sum.
   * Pre-condition: `n` is an integer; the loop executes only if `n > 2`.
   * Invariant: In each iteration, `ans` accumulates `i * (i+1)`.
   * Inline: `i * 1ll * (i + 1)` ensures that the multiplication `i * (i + 1)` is performed using `long long` arithmetic
   *         to prevent potential integer overflow before addition, especially for larger values of `i`.
   */
  for (int i = 2; i < n; ++i) {
    ans += i * 1ll * (i + 1);
  }
  // Block Logic: Prints the final calculated sum.
  cout << ans;

  return 0;
}