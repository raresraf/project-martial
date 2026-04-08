/**
 * @file 898B.CPP
 * @brief Solution to a variation of the Linear Diophantine Equation problem from Codeforces (Problem 898B).
 * Problem Statement: Determine if a given integer `n` can be represented as the sum `x*a + y*b`,
 *                    where `x` and `y` are non-negative integers. If possible, output "YES" and one
 *                    such pair `(x, y)`; otherwise, output "NO".
 * Algorithm: Iterative search. The algorithm iterates through all possible non-negative integer
 *            values for `x` (from `0` up to `n/a`), and for each `x`, it checks if the remaining
 *            value (`n - x*a`) is a non-negative multiple of `b`.
 * Time Complexity: O(N/a) in the worst case, where N is the value of `n` and `a` is the coefficient.
 * Space Complexity: O(1) for constant extra variables.
 * Domain: Competitive Programming, Number Theory, Brute Force Search.
 */
#include <bits/stdc++.h>
/**
 * @brief Macro to optimize C++ standard input/output operations.
 * Functional Utility: Disables synchronization between C-style I/O and C++ streams,
 *                     and unties `cin` from `cout`, leading to faster I/O operations
 *                     crucial for competitive programming.
 */
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
// Variables:
// idx: Unused, possibly from a template.
// x: Stores the found non-negative integer coefficient for 'a'. Initialized to -1 to indicate no solution found yet.
// y: Stores the found non-negative integer coefficient for 'b'.
// n: The target number to be represented as x*a + y*b.
// a: The first coefficient.
// b: The second coefficient.
long long idx = 0, x = -1, y, n, a, b;
/**
 * @brief Main function to solve the Linear Diophantine Equation variant.
 * Functional Utility: Reads the input values `n`, `a`, `b`, searches for a non-negative
 *                     integer solution `(x, y)`, and prints the result.
 */
int main() {
  optimizar_io // Functional Utility: Apply I/O optimizations.
  // Block Logic: Reads the target number 'n' and coefficients 'a' and 'b'.
  cin >> n >> a >> b;
  // Block Logic: Iterates through possible non-negative values for 'x' (represented by 'i').
  // The loop continues as long as 'i * a' does not exceed 'n', ensuring 'n - (i * a)' is non-negative.
  for (int i = 0; i * a <= n; ++i) {
    // Block Logic: Checks if the remaining value (n - i*a) is perfectly divisible by 'b'.
    // If it is, a non-negative 'y' exists, and a solution is found.
    if ((n - (i * a)) % b == 0) {
      x = i;                            // Assign the current 'i' as a valid 'x'.
      y = (n - (i * a)) / b;            // Calculate the corresponding 'y'.
      // Invariant: Once a solution (x, y) is found, the loop could technically break here
      // if only one solution is required. As currently implemented, it continues
      // to find the largest x that satisfies the condition for non-negative y.
    }
  }
  // Block Logic: Outputs the result based on whether a solution was found.
  // If 'x' is still -1, no solution was found.
  if (x == -1)
    cout << "NO\n";
  // If 'x' has been updated, a solution was found, and 'x' and 'y' are printed.
  else
    cout << "YES\n" << x << " " << y << "\n";
}
