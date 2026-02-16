/**
 * @file 670A.CPP
 * @brief Solution for a competitive programming problem involving calculating two values based on an input integer 'n', typically related to day counts or similar cyclical patterns.
 *
 * Algorithm: The problem likely involves calculating the minimum and maximum possible counts of certain events (e.g., weekends) given a total number of days 'n'. The solution uses integer division and modulo operations with 7 (representing days of the week) to determine a base count and then adjusts it based on the remainder.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as it uses a few constant-size variables.
 */

#include <bits/stdc++.h>
//! @brief Macro to optimize C++ standard I/O streams.
//! Functional Utility: Disables synchronization between C-style I/O (`stdio`) and C++ I/O (`iostream`)
//!                     and unties `cin` from `cout`. This typically results in faster input/output
//!                     operations, which is beneficial for competitive programming problems
//!                     where I/O speed can be a bottleneck.
#define OPTIMASI                                                               \
  cin.tie(0);                                                                  \
  cout.tie(0);                                                                 \
  ios_base::sync_with_stdio(0);
typedef long long int lli;
using namespace std;

/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Reads an integer `n` and calculates two derived values
 *                     based on `n`'s relationship with multiples of 7. It then
 *                     prints these two values separated by a space, followed by a newline.
 *                     The logic likely corresponds to specific problem constraints
 *                     related to weekly cycles or similar patterns.
 *
 * @return 0 on successful execution.
 */
int main() {
  OPTIMASI // Functional Utility: Apply I/O optimizations for faster execution.

  long long int n; // @brief Input integer representing the total count (e.g., total days).
  long long int k; // @brief Base count derived from `n` divided by 7 (e.g., full weeks).
  cin >> n;

  // Block Logic: Calculate the number of full weeks (`k`) and the remaining days (`n`).
  // Invariant: After these operations, `k` holds the integer quotient of `n / 7`, and
  //            `n` holds the remainder `n % 7`.
  k = n / 7;
  n = n - k * 7;

  // Block Logic: Determine and print the first calculated value.
  // This value typically represents a minimum or a specific boundary case.
  // Invariant: The output depends on whether the remainder `n` is 6.
  if (n == 6)
    cout << k * 2 + 1 << " "; // Functional Utility: Output adjusted for `n = 6`.
  else
    cout << k * 2 << " "; // Functional Utility: Output for other values of `n`.
  
  // Block Logic: Determine and print the second calculated value.
  // This value typically represents a maximum or another specific boundary case.
  // Invariant: The output depends on whether the remainder `n` is greater than 2.
  if (n > 2)
    cout << k * 2 + 2; // Functional Utility: Output adjusted for `n > 2`.
  else {
    cout << k * 2 + n; // Functional Utility: Output for `n <= 2`.
  }
  cout << endl; // Functional Utility: Print a newline character for proper output formatting.

  return 0;
}