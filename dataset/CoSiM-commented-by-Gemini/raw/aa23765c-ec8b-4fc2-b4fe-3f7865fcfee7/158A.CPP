/**
 * @file 158A.CPP
 * @brief An attempted solution for CodeForces Problem 158A - "Next Round".
 * @author Taro Young
 *
 * @warning The logic in this file is convoluted and incorrect for the general case.
 *          It appears to pass some test cases by coincidence or by exploiting properties
 *          of the test data, but it does not correctly implement the problem statement.
 *
 * @details
 * The "Next Round" problem requires counting the number of participants whose score is
 * positive and also greater than or equal to the score of the k-th ranked participant.
 *
 * Correct Algorithm:
 * 1. Read all `n` scores.
 * 2. Identify the cutoff score, which is `scores[k-1]`.
 * 3. Initialize a counter to 0.
 * 4. Iterate from `i = 0` to `n-1`: if `scores[i] >= cutoff_score` and `scores[i] > 0`, increment the counter.
 * 5. The result is the final value of the counter.
 *
 * This implementation attempts a different approach by modifying the rank `k` itself,
 * which is not a robust solution.
 *
 * @time_complexity O(N) in the worst case for the while loops.
 * @space_complexity O(N) to store the scores.
 */
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k;
  int a[110]; // Using a fixed-size array.
  cin >> n >> k;
  
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  /**
   * @brief This block contains the flawed logic. It tries to find the number of
   * advancing participants by adjusting the rank `k` directly, rather than by counting.
   */

  // Case 1: The k-th participant has a positive score.
  if (a[k - 1] > 0) {
    /**
     * @flaw This loop attempts to handle ties at the k-th position. It increments `k`
     * as long as the person at rank `k+1` has the same score as the person at rank `k`.
     * The final value of `k` is then printed. This only works by coincidence in cases
     * where all people with scores equal to the k-th score are contiguous and at the
     * end of the advancing list. It fails to count participants with higher scores
     * correctly and is not a general solution.
     */
    while (k < n && a[k] == a[k - 1]) {
      k++;
    }
  } else { // Case 2: The k-th participant has a score of 0 or less.
    /**
     * @brief This loop correctly finds the number of participants with positive scores
     * in the top `k` ranks. It decrements `k` until it finds a participant with a
     * positive score or `k` becomes 0. The final `k` represents the count of
     * participants with scores > 0.
     */
    while (k > 0 && a[k - 1] <= 0) {
      k--;
    }
  }

  // The output is the modified `k`, which is not the correct way to solve the problem generally.
  cout << k << endl;
  return 0;
}
