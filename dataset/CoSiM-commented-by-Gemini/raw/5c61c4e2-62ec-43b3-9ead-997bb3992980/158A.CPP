/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem from Codeforces.
 * @details The problem requires counting the number of participants who will advance
 * to the next round. A participant advances if their score is equal to or greater
 * than the score of the k-th place participant, provided they have a positive score.
 *
 * @algorithm Array Processing and Counting
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array.
 * 3. Identify the score of the k-th place participant (at index `k-1`).
 * 4. Iterate through the array of scores again.
 * 5. For each score, check if it is greater than or equal to the k-th score AND
 *    if it is positive.
 * 6. Increment a counter for each participant who meets both criteria.
 * 7. Print the final counter.
 *
 * @complexity
 * - Time: O(N), where N is the number of participants. The program iterates through
 *   the scores twice in separate loops.
 * - Space: O(N), to store the scores of all participants in an array.
 */

#include <iostream>

using namespace std;

int main(int argc, const char *argv[]) {
  // n: the total number of participants.
  // k: the rank of the participant whose score is the threshold.
  // count: the number of participants who will advance.
  int n, k, count;
  cin >> n;
  cin >> k;

  // `scores` will store the score of each participant.
  // Note: This is a Variable Length Array (VLA), which is a non-standard C++ extension.
  int scores[n];
  
  /**
   * @brief Read all participant scores into the array.
   */
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }

  // Initialize the counter for advancing participants.
  count = 0;

  /**
   * @brief Count the number of participants who advance.
   */
  // Invariant: `count` holds the number of advancing participants from the first `i` scores checked.
  for (int i = 0; i < n; i++) {
    /**
     * @brief Check if the current participant qualifies.
     * The conditions for advancing are:
     * 1. The score must be greater than or equal to the k-th participant's score.
     *    `scores[k - 1]` is used because arrays are 0-indexed.
     * 2. The score must be positive.
     */
    // Pre-condition: Check if the score meets the advancement criteria.
    if (scores[i] >= scores[k - 1] && scores[i] > 0)
      count++;
  }
  
  // Print the final count of advancing participants.
  cout << count;

  return 0; // Indicate successful execution.
}
