/**
 * @file 1366A.CPP
 * @brief Solution to a problem involving resource allocation or an optimization calculation
 *        based on two input integers.
 *
 * This program takes an integer 't' representing the number of test cases.
 * For each test case, it reads two integers, 'a' and 'b'. The objective is to
 * compute a specific result 'ans' derived from 'a' and 'b'. The calculation
 * involves ensuring 'a' is the smaller value, and then taking the minimum
 * of 'a' and one-third of the sum of 'a' and 'b'. This logic is typical
 * for problems requiring optimal distribution or a constraint-based maximum.
 *
 * Algorithm: Iterative processing of test cases, each involving input normalization
 *            (swapping to ensure order) and a specific mathematical calculation.
 * Time Complexity: O(T), where T is the number of test cases, as each test case
 *                  involves a constant number of operations.
 * Space Complexity: O(1) for storing variables.
 */
#include <algorithm> // Required for std::swap and std::min.
#include <iostream>  // Required for std::cin and std::cout.
using namespace std; // Using the standard namespace.

int t; // Global declaration for the number of test cases.
int a, b; // Global declarations for the two input integers per test case.

int main() {
  // Read the total number of test cases 't'.
  cin >> t;

  /**
   * Block Logic: Loop through each test case.
   * Invariant: 't' decreases with each iteration; 'ans' is calculated and printed
   *            for each pair of 'a' and 'b'.
   */
  while (t--) {
    int ans = 0; // Initialize 'ans' for the current test case.

    // Read the two integers 'a' and 'b' for the current test case.
    cin >> a >> b;

    /**
     * Block Logic: Ensures that 'a' always holds the smaller value and 'b' holds the larger value.
     * Pre-condition: 'a' and 'b' are read.
     */
    if (a > b) {
      // Swaps the values of 'a' and 'b' if 'a' is initially greater than 'b'.
      swap(a, b);
    }

    // Core Calculation: 'ans' is the minimum of 'a' and one-third of the sum (a + b).
    // This represents a key constraint or optimization criterion in the problem.
    ans = min(a, (a + b) / 3);

    // Print the calculated result for the current test case, followed by a newline.
    cout << ans << endl;
  }
  // The 'main' function implicitly returns 0 in C++ if no return statement is present.
}
