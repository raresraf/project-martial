/**
 * @file 1343C.CPP
 * @brief Solution to a competitive programming problem (Codeforces 1343C - "Alternating Subsequence").
 *
 * This program aims to find the maximum possible sum of an alternating subsequence.
 * An alternating subsequence consists of numbers where the signs of consecutive elements differ.
 * For each contiguous block of numbers with the same sign, the strategy is to select
 * the element with the maximum absolute value (i.e., the largest positive or smallest negative).
 *
 * Algorithm: Greedy approach with a single pass.
 * Iterates through the input sequence, keeping track of the maximum value within the current
 * contiguous block of numbers sharing the same sign. When the sign of the next number
 * changes or the end of the sequence is reached, the stored maximum for the current block
 * is added to the total sum, and a new block processing begins.
 * Time Complexity: O(N) per test case, where N is the length of the sequence. Total O(T * N).
 * Space Complexity: O(1) (excluding input storage).
 */
#include <iostream>
using namespace std;
#define ll long long

/**
 * @brief Main function: Entry point of the program.
 *
 * Handles multiple test cases, processing each sequence to find the maximum alternating subsequence sum.
 */
int main() {
  ll t, n, p, c, sum;

  // Pre-condition: Read the number of test cases.
  cin >> t;

  // Block Logic: Process each test case.
  // Invariant: 't' decreases with each iteration, loop continues until all test cases are processed.
  while (t--) {
    sum = 0; // Initialize sum for the current test case.
    // Pre-condition: Read 'n' (length of sequence) and the first element 'p'.
    cin >> n;
    cin >> p;
    n--; // Decrement 'n' as the first element has been read.
    // Block Logic: Iterate through the rest of the sequence.
    // Invariant: 'n' decreases with each iteration, loop continues until all elements are processed.
    while (n--) {
      // Pre-condition: Read the next element 'c'.
      cin >> c;
      // Block Logic: Check if the current element 'c' has the same sign as 'p'.
      // If p*c > 0, both are positive or both are negative.
      if (p * c > 0) {
        // If signs are the same, keep the maximum absolute value.
        // For positive numbers, we want the largest. For negative numbers, we want the largest (closest to zero).
        // Since we want the largest sum, we always pick the largest positive or smallest (closest to zero) negative.
        if (c > p)
          p = c;
      } else {
        // If signs are different (or one is zero), add the accumulated maximum 'p' to the total sum.
        sum += p;
        p = c; // Start a new alternating block with the current element 'c'.
      }
    }
    // Add the last accumulated maximum 'p' to the total sum and print the result.
    cout << sum + p << endl;
  }

  return 0;
}