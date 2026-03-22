/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A - Next Round.
 *        This program counts the number of participants who advance to the next round
 *        of a competition. Participants advance if their score is greater than or equal
 *        to the k-th place finisher's score, and their score is positive. It is assumed
 *        that participants' scores are given in non-increasing order.
 *
 * Algorithm: Iterative Scan with Early Exit
 *   1. Reads `n` (total participants) and `k` (the k-th place).
 *   2. **First Pass (for the first `k` participants):**
 *      a. Reads each participant's score.
 *      b. If a score of 0 is encountered, it means all subsequent scores are also
 *         0 or negative. Thus, `i` participants (those processed so far) advance.
 *         Prints `i` and exits.
 *   3. **Second Pass (for participants from k-th place onwards):**
 *      a. Reads each participant's score.
 *      b. Compares the current score with the `k`-th place score (which is stored in `a`).
 *      c. If the current score is different from the `k`-th place score, it implies
 *         subsequent scores will be lower (due to non-increasing order). Prints `i`
 *         (participants processed up to this point) and exits.
 *   4. If both loops complete without an early exit, it means all `n` participants
 *      meet the criteria. Prints `n`.
 *
 * Time Complexity: O(N), where N is the total number of participants, as the code
 *                  iterates through the scores at most once.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Next Round" problem.
   *
   * Reads the number of participants (`n`) and the k-th place (`k`), then
   * determines how many participants will advance based on their scores.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n, k; // `n`: total participants, `k`: the k-th place.
  cin >> n >> k;

  int a; // Stores the score of the k-th place participant.
  /**
   * Block Logic: Processes the scores of the first `k` participants.
   * Invariant: `i` represents the count of participants whose scores have been processed.
   * Pre-condition: `n` and `k` are positive integers.
   */
  for (int i = 0; i < k; i++) {
    cin >> a; // Reads the score of the current participant.
    /**
     * Conditional Logic: If a participant has a score of 0, no one after them
     *                    (including themselves) can advance (since scores are non-increasing).
     */
    if (a == 0) {
      cout << i << endl; // Prints the number of participants who advanced (those before this 0-scorer).
      return 0;           // Exits the program early.
    }
  }

  int b; // Stores the score of participants from k-th place onwards.
  /**
   * Block Logic: Processes the scores of participants from the k-th place to the end.
   * Compares each score to the k-th place score (`a`).
   * Invariant: `i` represents the count of participants whose scores have been processed.
   * Pre-condition: `a` holds the k-th place score.
   */
  for (int i = k; i < n; i++) {
    cin >> b; // Reads the score of the current participant.
    /**
     * Conditional Logic: If the current participant's score (`b`) is different
     *                    from the k-th place score (`a`), it means all subsequent
     *                    scores are lower, so no more participants with a high enough
     *                    score will be found.
     */
    if (b != a) {
      cout << i << endl; // Prints the number of participants who advanced.
      return 0;           // Exits the program early.
    }
  }
  // If the loops complete, it means all `n` participants have scores
  // greater than or equal to the k-th place and are positive.
  cout << n << endl; // Prints the total number of participants.
  return 0;
}
