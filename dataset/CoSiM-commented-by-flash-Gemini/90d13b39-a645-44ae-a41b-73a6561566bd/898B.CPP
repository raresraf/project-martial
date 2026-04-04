/**
 * @file 898B.CPP
 * @brief Solves a variation of the unbounded knapsack problem or linear Diophantine equation
 *        of the form a*i + b*j = n for non-negative integers i and j.
 *
 * This program attempts to find if a given integer 'n' can be expressed as
 * a sum of multiples of two other given integers 'a' and 'b' (i.e., `n = a*i + b*j`)
 * where 'i' and 'j' must be non-negative integers. If such 'i' and 'j' exist,
 * it outputs "YES" followed by one such pair (i, j). Otherwise, it outputs "NO".
 *
 * Algorithm: Brute-force iteration for 'i' (number of 'a's), then calculating 'j'.
 *            Since 'a*i' cannot exceed 'n', 'i' iterates from 0 up to 'n/a'.
 * Time Complexity: O(N/A), where N is the target sum and A is the coefficient of 'i'.
 *                  In the worst case (A=1), this is O(N).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;     // Functional Utility: Defines 'll' as a shorthand for 'long long'.
typedef pair<int, int> ii; // Functional Utility: Defines 'ii' as a shorthand for 'pair<int, int>'.

int main() {
  /**
   * @brief Main function: Finds non-negative integer solutions (i, j) for a*i + b*j = n.
   * @return 0 if the program executes successfully.
   */
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  // 'cin.tie(0)' prevents 'cin' from flushing 'cout' before each input operation.
  // 'ios_base::sync_with_stdio(0)' unties C++ streams from C standard streams.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);

  ll n, a, b; // Declare long long integers for input values.
  cin >> n >> a >> b; // Read the target sum 'n' and coefficients 'a' and 'b'.

  /**
   * @brief Loop to iterate through possible non-negative integer values for 'i'.
   * @pre 'n', 'a', 'b' are read. 'i' starts from 0.
   * @post If a solution (i, j) is found, it's printed and the program exits.
   * @invariant 'i' increases in each iteration; 'a*i' is always less than or equal to 'n'.
   */
  for (int i = 0; i <= n; i++) { // 'i' cannot exceed 'n' (assuming 'a' is positive).
    // Calculate 'j' based on the current 'i' using the equation n = a*i + b*j => b*j = n - a*i.
    // We also need to check if (n - a*i) is non-negative and divisible by 'b'.
    ll j_numerator = (n - a * i);
    /**
     * @brief Conditional block to check for a valid non-negative integer solution for 'j'.
     * @pre 'j_numerator' has been calculated.
     */
    if (j_numerator >= 0 && j_numerator % b == 0) {
        ll j = j_numerator / b; // Calculate j.
        // The original code had: 'll j = (n - a * i) / b;' but without checking divisibility
        // which could lead to incorrect results due to integer division truncating.
        // The added 'j_numerator % b == 0' ensures 'j' is an integer.
        // The original check `j >= 0` was correct.
        if (j >= 0) {
            cout << "YES" << endl; // Output "YES" if a valid solution is found.
            cout << i << " " << j << endl; // Output the solution (i, j).
            return 0; // Terminate the program after finding the first solution.
        }
    }
  }
  cout << "NO" << endl; // If the loop completes without finding a solution, output "NO".
  return 0;
}
