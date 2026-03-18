/**
 * @file 158A.CPP
 * @brief Solution for a programming contest problem to count participants advancing to the next round.
 *
 * Algorithm:
 * The program determines how many participants will advance based on a threshold score. The threshold
 * is the score of the k-th participant. A participant advances if their score is greater than or
 * equal to this threshold. The code includes a specific check that if the first participant's score
 * is 0, the count is immediately 0. This suggests a rule that scores must be positive, but this
 * rule is not explicitly or correctly enforced in the main counting loop for all cases. The code
 * is documented as it is written, preserving its original logic.
 *
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N), for storing the scores of the N participants in an array.
 */
#include <iostream>
using namespace std;
int main() {
  // a: array to store the scores of up to 50 participants.
  int a[50];
  // coun: counter for participants who will advance.
  // n: total number of participants.
  // i, j: loop counters.
  // k: the rank of the participant whose score is the threshold.
  int coun = 0, n, i, k;
  
  // Read the number of participants (n) and the rank for the threshold score (k).
  cin >> n >> k;
  
  /**
   * @brief Reads the scores of all n participants into the array 'a'.
   * Pre-condition: n is initialized with the number of participants.
   */
  for (i = 1; i <= n; i++) {
    cin >> a[i - 1];
  }
  
  // val: stores the threshold score, which is the score of the k-th participant (0-indexed).
  int val = a[k - 1];
  
  /**
   * @brief This block implements the logic to count advancing participants.
   * It first handles a specific edge case: if the first participant's score is 0, it assumes
   * no one advances. This is likely an incomplete attempt to handle a "scores must be positive" rule.
   */
  if (a[0] == 0)
    cout << "0";
  else {
    /**
     * @brief Main counting loop. It counts any participant whose score is greater than or
     * equal to the k-th participant's score.
     * Invariant: 'coun' holds the number of participants found so far who meet the threshold.
     */
    for (int j = 0; j < n; j++) {
      if (a[j] >= val)
        coun++;
    }
    cout << coun;
  }

  return 0;
}
