/**
 * @file 158A.CPP
 * @brief Solution for a competitive programming problem involving score-based qualification.
 *
 * This program determines how many participants will advance to the next round based on their scores.
 * The rule for advancing is that a participant's score must be positive and must be
 * equal to or greater than the score of the k-th place participant.
 *
 * @algorithm
 * 1. Read the total number of participants (n) and the rank (k) to be used as the benchmark.
 * 2. Iterate through the participants' scores, reading them one by one.
 * 3. The first 'k' participants who have a positive score are provisionally advanced. The k-th participant's score is stored as the benchmark score.
 * 4. For subsequent participants (after the k-th), they advance only if their score is positive and equal to the benchmark score.
 * 5. The iteration stops if a participant has a score of 0, or if a participant after the k-th position has a score lower than the benchmark.
 * 6. The final count of advanced participants is printed.
 *
 * @complexity
 * Time Complexity: O(n), as the program iterates through the list of participants' scores once.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: total number of participants, k: the rank of the benchmark score.
  int n, k;
  // a: stores the score of the current participant.
  int a;
  // current: stores the score of the k-th participant or the last participant to qualify within the top k.
  int current;
  // result: counter for the number of participants who advance.
  int result = 0;

  // Read the number of participants and the benchmark rank.
  cin >> n >> k;

  /**
   * @brief Main loop to process scores.
   * Iterates through all 'n' participants to count who qualifies.
   */
  for (int i = 0; i < n; i++) {
    // Read the score of the current participant.
    cin >> a;

    // Pre-condition: Any participant with a score of 0 cannot advance.
    // Since scores are given in non-increasing order, no subsequent participant can advance either.
    if (a == 0) {
      break;
    }

    /**
     * @brief Logic for advancing participants.
     * The first 'k' participants with positive scores will advance.
     */
    if (result < k) {
      result++;
      // Store the score of the k-th placer (or the last person to be counted if n < k).
      current = a;
    } else {
      // Logic for participants after the k-th position.
      // They must have the same score as the k-th placer to advance.
      if (current == a) {
        result++;
      } else {
        // If the score is not the same (and since it's non-increasing, it must be lower),
        // no more participants can qualify.
        break;
      }
    }
  }

  // Output the total number of participants who advanced.
  cout << result;
  return 0;
}
