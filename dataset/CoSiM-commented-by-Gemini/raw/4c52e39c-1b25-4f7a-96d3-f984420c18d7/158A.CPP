/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details The problem requires counting the number of participants who will
 * advance to the next round. A participant advances if their score is
 * greater than or equal to the score of the k-th place finisher, and their
 * score is positive. The input scores are provided in non-increasing order.
 *
 * @algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array.
 * 3. Identify the threshold score, which is the score of the k-th participant
 *    (at index `k-1`).
 * 4. Iterate through all participants' scores.
 * 5. For each score, check if it is greater than or equal to the threshold
 *    score AND if it is positive.
 * 6. Increment a counter for each participant who meets both criteria.
 * 7. Print the final count.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of participants, due to the
 * loops for reading and checking the scores.
 * Space Complexity: O(N) to store the scores in an array.
 */
#include <iostream>
using namespace std;

int main() {
  // n: the total number of participants.
  // k: the rank of the participant whose score is the threshold.
  // a: an array to store the scores of the participants.
  // advancing_participants: a counter for those who advance.
  int n, k, a[100], advancing_participants = 0;

  // Read the number of participants and the k-th rank.
  cin >> n >> k;

  /**
   * @block
   * @brief Read all participant scores into an array.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  /**
   * @block
   * @brief Count the number of participants who advance.
   * @invariant Iterates through each of the `n` participants.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @block
     * @brief Check if the current participant advances.
     * @pre The score `a[i]` is the score of the current participant.
     *      The threshold score is `a[k-1]`.
     * @post The `advancing_participants` counter is incremented if the
     *       conditions are met.
     */
    if (a[i] >= a[k - 1] && a[i] > 0) {
      advancing_participants++;
    }
  }

  // Print the total count of advancing participants.
  cout << advancing_participants << endl;

  return 0;
}