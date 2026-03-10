/**
 * @file 1343C.CPP
 * @brief Solution for a competitive programming problem, likely "Alternating Subsequence".
 * The goal is to find an alternating subsequence (in terms of sign) that is as
 * long as possible, and among all such subsequences, find the one with the
 * maximum sum. This implementation attempts to solve this using a dynamic
 * programming-like approach.
 *
 * @algorithm A linear-time dynamic programming approach. The code maintains the
 * length and maximum sum of the best alternating subsequence found so far that
 * ends in a positive number (`posLen`, `maxPos`) and the best one that ends
 * in a negative number (`negLen`, `maxNeg`).
 *
 * @time_complexity O(N) for each test case, where N is the number of elements in the array.
 * @space_complexity O(1) as it processes the input stream element by element.
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  int t, n, v;
  scanf("%d", &t);
  while (t--) {
    scanf("%d", &n);
    ll maxNeg = 0, maxPos = 0; // DP state: max sum ending in neg/pos
    ll negLen = 0, posLen = 0; // DP state: max length ending in neg/pos

    /**
     * @block
     * @brief Iterates through the input, updating DP states.
     */
    for (int i = 0; i < n; i++) {
      scanf("%d", &v);
      if (v > 0) {
        /**
         * @brief Transition for a positive number `v`.
         * A positive number can extend a subsequence that ended in a negative number.
         * A new potential length and sum are calculated. The DP state for
         * positive-ending subsequences (`posLen`, `maxPos`) is updated if this
         * new subsequence is longer, or if it's the same length but has a larger sum.
         */
        ll nextLen = negLen + 1;
        ll nextVal = maxNeg + v;
        if (nextLen > posLen) {
          posLen = nextLen;
          maxPos = nextVal;
        } else if (nextVal > maxPos)
          maxPos = nextVal;
      } else {
        /**
         * @brief Transition for a negative number `v`.
         * A negative number can extend a subsequence that ended in a positive number.
         * The logic is symmetric to the positive case.
         */
        ll nextLen = posLen + 1;
        ll nextVal = maxPos + v;
        if (nextLen > negLen) {
          negLen = nextLen;
          maxNeg = nextVal;
        } else if (nextVal > maxNeg)
          maxNeg = nextVal;
      }
    }

    /**
     * @block
     * @brief Selects the best result from the two final DP states.
     * Compares the subsequence ending in a positive number with the one ending
     * in a negative number, prioritizing length first, then sum.
     */
    ll bestLen = negLen;
    ll bestVal = maxNeg;
    if (posLen > bestLen || (posLen == bestLen && maxPos > bestVal)) {
      bestLen = posLen;
      bestVal = maxPos;
    }
    printf("%lld\n", bestVal);
  }
}