/**
 * @file 753A.CPP
 * @brief This program generates a sequence of positive integers whose sum equals a given input 'n'.
 * The sequence aims to use as many distinct positive integers as possible, starting from 1.
 * If a remainder 'n' exists after forming the initial sequence, it is added to the last element.
 * The output includes the count of elements in the generated sequence and the sequence itself.
 *
 * Algorithm: Greedy sequence generation and remainder distribution.
 * Time Complexity: O(sqrt(N)) because the loop variable 'i' increases and 'n' decreases quadratically.
 * Space Complexity: O(N) to store the dynamic programming array `dp`.
 */

#include <bits/stdc++.h> // Includes iostream, etc.
using namespace std;

int main() {
  // Functional Utility: 'n' stores the target sum, which is decremented as elements are added to the sequence.
  int n;
  // Functional Utility: 'k' stores the number of elements in the generated sequence (one more than the last 'i').
  int k;
  // Functional Utility: 'i' is a counter used for generating sequence elements and as a loop iterator.
  int i;

  // Block Logic: Reads the target sum 'n' from standard input.
  cin >> n;

  // Functional Utility: 'dp' is a dynamic array (VLA - Variable Length Array) of size 'n' to store the generated sequence.
  //                     Initialized to zeros. VLAs are a C99 feature and a GCC extension in C++, not standard C++.
  int dp[n] = {};
  // Functional Utility: Initializes 'i' to 1, as the sequence starts with positive integers from 1.
  i = 1;

  /**
   * Block Logic: Greedily generates a sequence of distinct positive integers (1, 2, 3, ...)
   *              whose sum does not exceed the remaining 'n'.
   * Invariant: In each iteration, 'i' is added to the sequence and subtracted from 'n'.
   *            The loop continues as long as 'i' can be added without exceeding 'n'.
   */
  while (i <= n) {
    // Functional Utility: Stores the current integer 'i' in the 'dp' array as part of the sequence.
    dp[i] = i;
    // Functional Utility: Subtracts the current integer 'i' from the remaining target sum 'n'.
    n -= i;
    // Functional Utility: Increments 'i' to consider the next positive integer.
    i++;
  }
  /**
   * Block Logic: If there is a remaining value in 'n' after the loop, it's added to the last element of the sequence.
   * Pre-condition: The 'while' loop has terminated.
   * Invariant: Any remaining value of 'n' is distributed to the last element, ensuring the sum remains 'n'.
   */
  if (n) { // If 'n' is non-zero
    dp[i - 1] += n; // Add the remainder to the last element stored (at index i-1).
  }
  // Functional Utility: 'k' stores the total number of distinct elements considered (i-1 before the remainder).
  // The value of 'i' here is one more than the actual number of elements in the sequence before remainder handling.
  k = i;

  // Output the count of elements in the final sequence.
  // Inline: 'k - 1' correctly reflects the number of elements because 'i' was incremented one last time
  //         before the loop condition failed.
  cout << k - 1 << endl;

  /**
   * Block Logic: Prints the generated sequence of numbers.
   * Invariant: Iterates from the first element up to (but not including) 'k', printing each element.
   */
  for (i = 1; i < k; i++)
    cout << dp[i] << " ";
}
