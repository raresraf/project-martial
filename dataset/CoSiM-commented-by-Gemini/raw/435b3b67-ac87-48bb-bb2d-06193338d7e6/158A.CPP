/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A - Next Round.
 *
 * @algorithm
 * The problem is to count how many participants advance to the next round. The
 * rule is that a participant's score must be positive and greater than or equal
 * to the score of the k-th ranked participant.
 *
 * This solution uses a straightforward two-pass approach:
 * 1. Read all participant scores into an array.
 * 2. In a second pass, iterate through the array again, comparing each
 *    participant's score to the k-th participant's score.
 * 3. A counter is incremented for each participant who has a positive score
 *    that is also greater than or equal to the k-th score.
 *
 * @warning This code uses 1-based indexing for its loops and array access
 *          (e.g., `for (i=1; i<=n; ...)` and `a[k]`). C++ arrays are 0-indexed.
 *          This is a dangerous practice that can lead to out-of-bounds memory
 *          access and undefined behavior, although it may appear to work for
 *          small inputs within the declared array size. The k-th element in a
 *          0-indexed array should be at index `k-1`.
 *
 * @complexity
 * - Time Complexity: O(N), for the two separate passes over the N participants.
 * - Space Complexity: O(N), to store the scores in an array.
 */
#include <iostream>
using namespace std;
int main() {
  int n, k, i, counter = 0;
  int a[103];
  cin >> n >> k;
  /**
   * @brief Reads all n scores into the array `a`, using 1-based indexing.
   */
  for (i = 1; i <= n; i++) {
    cin >> a[i];
  }
  /**
   * @brief Counts the number of participants who qualify.
   * It compares each score `a[i]` against the k-th score `a[k]`.
   */
  for (i = 1; i <= n; i++) {
    if (a[i] >= a[k] && a[i] > 0) {
      counter++;
    }
  }
  cout << counter << endl;
}
