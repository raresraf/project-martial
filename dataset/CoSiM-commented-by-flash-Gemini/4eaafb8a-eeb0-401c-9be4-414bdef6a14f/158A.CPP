/**
 * @file 158A.CPP
 * @brief Counts participants advancing in a competition based on a k-th place cutoff score.
 *
 * Problem Name: Next Round (Codeforces 158A)
 * Algorithm: Iterative comparison. Reads all scores, determines the cutoff score from the k-th participant,
 *            then re-iterates to count participants meeting the advancement criteria.
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
  int arr[51]; // Array to store scores of up to 50 participants.
  int n, k, result = 0; // 'n': total participants, 'k': cutoff rank, 'result': count of advancing participants.

  // Functional Utility: Reads the total number of participants and the cutoff rank.
  cin >> n >> k;

  /**
   * Block Logic: Reads 'n' participant scores into the array 'arr'.
   * Invariant: After this loop, 'arr' contains all participant scores.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i]; // Reads the score for the i-th participant.
  }

  // Functional Utility: Adjusts 'k' to be a 0-based index for array access.
  k -= 1;
  // Functional Utility: Stores the cutoff score, which is the score of the k-th ranked participant.
  int val = arr[k];

  /**
   * Block Logic: Iterates through all participant scores to count those who advance.
   * Invariant: `result` increments for each participant whose score is greater than or equal to `val`
   *            (k-th place score) and also positive.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Checks if the current participant's score meets the advancement criteria.
     * Pre-condition: `arr[i]` is the current participant's score, `val` is the cutoff score.
     * Invariant: If `arr[i]` is greater than or equal to `val` AND `arr[i]` is positive, the participant advances.
     */
    if (arr[i] >= val && arr[i] != 0)
      result++; // Increments count of advancing participants.
  }

  // Functional Utility: Prints the final count of advancing participants.
  cout << result;
  // Indicates successful program execution.
  return 0;
}
