/**
 * @file 158A.CPP
 * @brief Counts participants advancing in a competition based on a k-th place cutoff score.
 *
 * Algorithm: Iterative comparison. Reads all scores, then iterates to check advancement conditions.
 * Time Complexity: O(N) where N is the number of participants, due to two linear passes.
 * Space Complexity: O(N) for storing participant scores.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to determine the number of advancing participants.
 * @pre Input consists of two integers: 'n' (total participants) and 'k' (cutoff rank),
 *      followed by 'n' integers representing participant scores.
 * @post Prints the total count of participants who advance to the next round.
 */
int main() {
  int n, k, a[100], t = 0; // 'n': total participants, 'k': cutoff rank, 'a': array of scores, 't': count of advancing participants.

  // Functional Utility: Reads the total number of participants and the cutoff rank.
  cin >> n >> k;

  /**
   * Block Logic: Reads 'n' participant scores into the array 'a'.
   * Invariant: After each iteration, 'a[i]' holds the score of the i-th participant.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  /**
   * Block Logic: Iterates through all participant scores to determine who advances.
   * Invariant: 't' increments for each participant whose score meets the advancement criteria
   *            (score >= k-th place score AND score > 0).
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Checks if the current participant's score meets the advancement criteria.
     * Pre-condition: 'a[i]' is the current participant's score, 'a[k-1]' is the k-th place score.
     * Invariant: If both conditions are true, the participant advances.
     */
    if (a[i] >= a[k - 1] && a[i] > 0) {
      t++; // Increment count of advancing participants.
    }
  }

  // Functional Utility: Prints the total count of advancing participants.
  cout << t << endl;

  // Indicates successful program execution.
  return 0;
}
