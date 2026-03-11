/**
 * @file 1140D.CPP
 * @brief Solution to a specific summation problem (likely Codeforces 1140D).
 *
 * This program calculates the sum of the series i * (i + 1) for 'i' ranging from 2 to 'n-1',
 * where 'n' is an integer input. The problem context typically involves optimal triangulation
 * or similar geometric/combinatorial tasks where this sum arises.
 * Algorithm: Direct summation by iterating from 2 to n-1 and accumulating i * (i + 1).
 * Time Complexity: O(N) where N is the input integer, as the loop runs approximately N times.
 * Space Complexity: O(1) - constant space used for variables.
 */

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming.
using namespace std;       // Uses the standard namespace for convenience.

/**
 * @brief Main function to compute the required summation.
 *
 * Reads an integer 'n', calculates the sum of i*(i+1) from i=2 to n-1,
 * and prints the final sum.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  // Block Logic: Read the input integer 'n'.
  // Pre-condition: User provides a single integer 'n'.
  // Invariant: 'n' holds the input value.
  cin >> n;

  long long ans = 0; // Initialize a long long variable to store the accumulated sum.
                     // Using long long prevents overflow for larger values of n.

  // Block Logic: Calculate the sum of i * (i + 1) for i from 2 to n-1.
  // This loop iteratively adds the product of 'i' and 'i+1' to 'ans'.
  // Pre-condition: 'n' is at least 3 for the loop to execute.
  // Invariant: 'ans' accumulates the correct sum up to the current iteration.
  for (int i = 2; i <= n - 1; i++) {
    ans += (i * (i + 1));
  }

  // Block Logic: Print the final calculated sum.
  // A newline character is added for proper formatting.
  // Post-condition: The sum is printed to standard output.
  cout << ans << "\n";
}