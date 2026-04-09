/**
 * @file 1343C.CPP
 * @brief This program solves a problem related to finding the maximum sum of a subsequence
 * with alternating signs (positive and negative numbers) from a given sequence of integers.
 * The subsequence should have the maximum possible sum. If multiple subsequences have the
 * same maximum sum, the one with the maximum length is preferred.
 *
 * Algorithm: Dynamic Programming (Greedy-like approach).
 * The algorithm iterates through the input array, maintaining two states:
 * `maxPos`: The maximum sum of an alternating subsequence ending with a positive number.
 * `posLen`: The length of the subsequence corresponding to `maxPos`.
 * `maxNeg`: The maximum sum of an alternating subsequence ending with a negative number.
 * `negLen`: The length of the subsequence corresponding to `maxNeg`.
 *
 * For each new number `v`:
 * - If `v` is positive: It can extend a subsequence ending in a negative number.
 *   Calculate a potential new sum (`maxNeg + v`) and length (`negLen + 1`).
 *   Update `maxPos` and `posLen` if this new subsequence is better (longer, then larger sum).
 * - If `v` is negative: It can extend a subsequence ending in a positive number.
 *   Calculate a potential new sum (`maxPos + v`) and length (`posLen + 1`).
 *   Update `maxNeg` and `negLen` if this new subsequence is better.
 *
 * Time Complexity: O(N) where N is the number of elements in the sequence, as it involves a single pass through the array.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */

#include <bits/stdc++.h>
using namespace std;
#define ll long long

/**
 * @brief Main function of the program.
 * Reads test cases, processes each sequence to find the maximum alternating subsequence sum.
 * @pre Input starts with an integer `t` (number of test cases). Each test case begins with `n`
 *      (number of elements in the sequence), followed by `n` integers.
 * @post For each test case, prints the maximum sum of an alternating subsequence.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: 't' for number of test cases, 'n' for sequence length, 'v' for current element.
  int t, n, v;
  // Functional Utility: Reads the number of test cases.
  scanf("%d", &t);
  // Block Logic: Processes each test case.
  // Invariant: At the start of each iteration, `t` decrements, indicating remaining test cases.
  while (t--) {
    // Functional Utility: Reads the number of elements for the current test case.
    scanf("%d", &n);
    // Functional Utility: `maxNeg` stores the maximum sum of an alternating subsequence ending with a negative number.
    // Functional Utility: `maxPos` stores the maximum sum of an alternating subsequence ending with a positive number.
    ll maxNeg = 0, maxPos = 0;
    // Functional Utility: `negLen` stores the length of the subsequence corresponding to `maxNeg`.
    // Functional Utility: `posLen` stores the length of the subsequence corresponding to `maxPos`.
    ll negLen = 0, posLen = 0;

    // Block Logic: Iterates through each element of the sequence to build alternating subsequences.
    // Invariant: `maxNeg`, `negLen`, `maxPos`, `posLen` are updated to reflect the best
    //            alternating subsequences found so far up to the current element.
    for (int i = 0; i < n; i++) {
      // Functional Utility: Reads the current element of the sequence.
      scanf("%d", &v);
      // Block Logic: If the current element is positive.
      // Pre-condition: `v` is the current element being processed.
      if (v > 0) {
        // Functional Utility: Calculates the potential length and sum if `v` extends a subsequence ending in a negative number.
        ll nextLen = negLen + 1;
        ll nextVal = maxNeg + v;
        // Block Logic: Updates `maxPos` and `posLen` if the new positive-ending subsequence is better.
        // Preference is given to longer subsequences, then to larger sums.
        if (nextLen > posLen) {
          posLen = nextLen;
          maxPos = nextVal;
        } else if (nextVal > maxPos)
          maxPos = nextVal;
      } else { // Block Logic: If the current element is negative.
        // Functional Utility: Calculates the potential length and sum if `v` extends a subsequence ending in a positive number.
        ll nextLen = posLen + 1;
        ll nextVal = maxPos + v;
        // Block Logic: Updates `maxNeg` and `negLen` if the new negative-ending subsequence is better.
        // Preference is given to longer subsequences, then to larger sums.
        if (nextLen > negLen) {
          negLen = nextLen;
          maxNeg = nextVal;
        } else if (nextVal > maxNeg)
          maxNeg = nextVal;
      }
    }

    // Functional Utility: `bestLen` and `bestVal` store the overall best length and sum found.
    ll bestLen = negLen;
    ll bestVal = maxNeg;
    // Block Logic: Compares the positive-ending and negative-ending subsequences to find the overall best.
    // Preference is given to longer subsequences, then to larger sums.
    if (posLen > bestLen || (posLen == bestLen && maxPos > bestVal)) {
      bestLen = posLen;
      bestVal = maxPos;
    }
    // Functional Utility: Prints the maximum sum of the alternating subsequence.
    printf("%lld
", bestVal);
  }
}
