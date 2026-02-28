/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem from Codeforces.
 * @details This program determines how many participants will advance to the next round
 * of a competition. A participant advances if they have a positive score that is
 * greater than or equal to the score of the k-th place participant.
 *
 * @algorithm Two-Pass Array Processing
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array.
 * 3. In a second pass, iterate through the array of scores.
 * 4. For each score, check if it meets two criteria:
 *    a) It is greater than or equal to the score at the k-th position (index `k-1`).
 *    b) It is a positive score (i.e., not zero).
 * 5. Count how many participants meet both criteria.
 * 6. Print the final count.
 *
 * @complexity
 * - Time: O(N), where N is the number of participants, due to two separate loops
 *   over the array of scores.
 * - Space: O(C), where C is a constant (100 in this case). The space complexity
 *   does not depend on N, as a fixed-size array is used.
 */

#include <iostream>
using namespace std;

int main() {
  // n: the number of participants.
  // k: the rank that serves as the score threshold.
  int n, k;
  cin >> n;
  cin >> k;
  
  // `a` is a fixed-size array to store participant scores.
  // Note: This assumes `n` will not exceed 100.
  int a[100];
  
  /**
   * @brief Read all participant scores into the array.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  
  // `c` is the counter for participants who advance to the next round.
  int c = 0;
  
  /**
   * @brief Count the number of participants who qualify.
   */
  // Invariant: `c` holds the count of advancing participants from the first `i` scores.
  for (int i = 0; i < n; i++) {
    /**
     * @brief Check if the current participant's score qualifies them to advance.
     * The score must be positive and at least as high as the k-th participant's score.
     */
    // Pre-condition: Check if score is high enough and positive.
    if (a[i] >= a[k - 1] && a[i] != 0) {
      c++;
    }
  }
  
  // Print the final count.
  cout << c;

  return 0; // Indicate successful execution.
}
