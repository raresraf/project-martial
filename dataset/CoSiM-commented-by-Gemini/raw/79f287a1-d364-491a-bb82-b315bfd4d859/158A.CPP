/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A, "Next Round".
 * @details This program calculates the number of participants who will advance
 * to the next round of a competition. A participant advances if they achieve a
 * positive score and their score is equal to or greater than the score of the
 * k-th place finisher.
 *
 * Algorithm:
 * 1. Read the total number of participants `n` and the rank cutoff `k`.
 * 2. Iterate through the first `k` participants. Count them as long as their
 *    score is positive. If a score of 0 is encountered, no one from that
 *    point on can qualify, so the process stops and the current count is the
 *    final answer.
 * 3. After the first `k` participants are checked (assuming all had positive
 *    scores), the score of the k-th participant is held in the `cur` variable.
 * 4. Continue reading the scores of the remaining participants (from k+1 to n).
 * 5. Count these additional participants as long as their score is positive
 *    (guaranteed by the problem constraints and the fact that `cur > 0`) and
 *    equal to the k-th place score.
 * 6. Print the total count of advancing participants.
 *
 * Time Complexity: O(N), where N is the number of participants, as it iterates
 * through the list of scores once.
 * Space Complexity: O(1).
 */
#include <iostream>
using std::cin;
using std::cout;
int main() {
  int n, k, cur = 0, ct = 0, rest;
  cin >> n >> k;
  /**
   * @brief This loop processes the first `k` participants. It counts them as
   * advancing (`ct`) as long as their scores are positive.
   * The loop uses a `++i, ++ct` construct, which increments both the loop
   * counter and the number of advancing participants simultaneously.
   */
  for (int i = 0; i < k; ++i, ++ct) {
    cin >> cur;
    /**
     * @brief If a participant within the top k has a score of 0, no one
     * from this rank or lower can advance. The loop terminates, and the
     * current count `ct` (which is `i` at this point) is printed.
     */
    if (!cur) {
      cout << ct;
      return 0;
    }
  }
  /**
   * @brief This loop continues from the (k+1)-th participant. It counts any
   * additional participants who tie with the k-th place finisher's score.
   * The loop condition `cur == rest` ensures only ties are counted.
   */
  while (cin >> rest && cur == rest) {
    ++ct;
  }
  cout << ct;
  return 0;
}
