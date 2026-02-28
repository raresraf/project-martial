/**
 * @file 1343C.CPP
 * @brief Solution to a competitive programming problem (Codeforces 1343C - "Alternating Subsequence").
 *
 * This program calculates the maximum possible sum of an alternating subsequence from a given array.
 * An alternating subsequence means that the signs of consecutive elements are different
 * (e.g., positive, negative, positive, or negative, positive, negative).
 * The strategy is to select, for each contiguous block of same-signed numbers,
 * the element with the maximum absolute value (largest positive or negative closest to zero).
 *
 * Algorithm: Dynamic Programming.
 * This implementation uses a dynamic programming approach with two states to keep track
 * of the maximum sum ending with a positive or a negative number, and two auxiliary
 * states to keep track of the maximum element found within the current contiguous block
 * of positive or negative numbers.
 * `a[0]` stores the current maximum negative number in a consecutive negative segment.
 * `a[1]` stores the current maximum positive number in a consecutive positive segment.
 * `dp[0]` stores the maximum alternating sum ending with a negative number.
 * `dp[1]` stores the maximum alternating sum ending with a positive number.
 *
 * Time Complexity: O(N) for each test case, where N is the number of elements in the sequence.
 * Total time complexity is O(T * N), where T is the number of test cases.
 * Space Complexity: O(1) because the `a` and `dp` arrays have a fixed size of 2.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
#define FOR(i, a, b) for (int i = (a); i < (b); i++) // Macro for a for-loop.
#define ll long long                          // Type alias for long long.
#define N 100010                              // Constant for maximum N, not directly used in this code's arrays.
using namespace std;

ll t, n, x, a[2], dp[2]; // 't': test cases, 'n': sequence length, 'x': current element,
                         // 'a[2]': max element in current positive/negative block,
                         // 'dp[2]': max alternating sum ending with positive/negative.

/**
 * @brief Main function: Entry point of the program.
 *
 * Processes multiple test cases. For each test case, it reads a sequence of numbers
 * and computes the maximum alternating subsequence sum using a dynamic programming approach.
 */
int main() {
  cin >> t; // Pre-condition: Read the number of test cases.
  // Block Logic: Process each test case.
  // Invariant: 't' decreases with each iteration, loop continues until all test cases are processed.
  while (t--) {
    cin >> n; // Pre-condition: Read 'n', the length of the current sequence.
    // Initialize 'a' arrays to INT_MIN to signify no number seen yet for that sign.
    a[0] = a[1] = INT_MIN;
    // Initialize 'dp' arrays to 0.
    dp[0] = dp[1] = 0;
    // Block Logic: Iterate through each number in the sequence.
    // Invariant: 'i' represents the current index in the input sequence.
    FOR(i, 0, n) {
      cin >> x; // Pre-condition: Read the current number 'x'.

      // Functional Utility: Reset the 'a' value for the *opposite* sign.
      // If x is negative (x<0 is true, index 1), reset a[1] (max positive).
      // If x is positive (x<0 is false, index 0), reset a[0] (max negative).
      // This signifies that a new block of the current sign is starting or continuing,
      // and we cannot extend the previous block of the opposite sign with this 'x'.
      a[x < 0] = INT_MIN;

      // Block Logic: Update dp and a for the current number 'x'.
      // x > 0 is true (1) for positive, false (0) for negative.
      // This effectively maps positive numbers to index 1 and negative to index 0.

      // If 'a' for the current sign is INT_MIN, it means a new segment of this sign begins.
      // This is the first number of this sign encountered in the current sequence.
      if (a[x > 0] == INT_MIN) {
        // The maximum sum ending with 'x' (current sign) is the sum ending with
        // the opposite sign's best sequence plus 'x'.
        dp[x > 0] = dp[x < 0] + x;
      } else {
        // If 'a' for the current sign is not INT_MIN, it means we are continuing a segment
        // of numbers with the same sign. We keep the maximum value in this segment.
        a[x > 0] = max(a[x > 0], x); // Update `a` to store the largest value in this segment.
        // Update the max alternating sum ending with this sign.
        // It's either the current dp value or extending the opposite sign's best sequence
        // with the current best (largest) element of this segment.
        dp[x > 0] = max(dp[x > 0], dp[x < 0] + a[x > 0]);
      }
      // This line appears redundant given the previous update in the else block.
      // However, its presence ensures `a[x > 0]` is updated if it was `INT_MIN` originally.
      // It effectively ensures `a[x > 0]` always holds the maximum seen in the current segment of its sign.
      a[x > 0] = max(a[x > 0], x);
    }
    // Output the maximum alternating subsequence sum.
    // The final result is the maximum sum ending with either a positive or negative number.
    // Given the problem constraints and how `dp` is built, `dp[x>0]` (final sign) should contain the overall max.
    cout << dp[x > 0] << '\n';
  }

  return 0; // Indicate successful execution.
}