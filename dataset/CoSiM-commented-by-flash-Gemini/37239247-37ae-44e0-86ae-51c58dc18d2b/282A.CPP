/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - Bit++.
 *
 * This program simulates a simple counter operation. It reads a series of statements,
 * each either incrementing or decrementing a single variable 'X'. The goal is to
 * determine the final value of 'X' after all operations.
 *
 * Algorithm: Direct conditional update. The program parses the middle character of
 * the input string to decide whether to increment or decrement the counter.
 * Time Complexity: O(T * L), where T is the number of test cases (operations) and L is the
 *                  maximum length of an input string (constant, typically 3).
 * Space Complexity: O(L) for storing the input string.
 */

// link : https://codeforces.com/problemset/problem/282/A

#include <bits/stdc++.h>

using namespace std;

string s; // Stores the current operation string (e.g., "++X", "X--").
int t;    // Stores the number of operations to perform.
int k;    // The counter variable, initialized to 0 implicitly.

main() {
  /**
   * @brief Main function to execute the Bit++ problem logic.
   *
   * Reads the number of operations, then processes each operation to update
   * a counter. Finally, prints the accumulated result.
   */

  // Read the total number of operations.
  cin >> t;

  // Block Logic: Iterates through each operation provided in the input.
  // Invariant: 't' decreases with each iteration, and the loop continues as
  // long as there are pending operations.
  while (t--) {

    // Read the operation string for the current iteration.
    cin >> s;

    // Conditional logic to determine if 'k' should be decremented or incremented.
    // The problem statement implies that the second character ('s[1]') is either
    // '+' or '-'.
    if (s[1] == '-')
      k--; // Decrement 'k' if the operation indicates subtraction.

    else
      k++; // Increment 'k' if the operation indicates addition.
  }

  // Output the final value of the counter 'k'.
  cout << k;
}
