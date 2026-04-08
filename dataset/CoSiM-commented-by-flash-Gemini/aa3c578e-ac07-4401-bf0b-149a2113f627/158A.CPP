/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces (Problem 158A).
 * @details This program calculates the number of participants who will advance to the next round.
 * A participant advances if their score is equal to or greater than the k-th place finisher's
 * score, and their score is positive.
 *
 * @algorithm
 * 1. Read the total number of participants, `n`, and the rank `k`.
 * 2. Read the scores of all `n` participants into an array.
 * 3. The score of the k-th participant (at index `k-1`) is the threshold score.
 * 4. Iterate through the scores of all participants.
 * 5. For each participant, check if their score is positive AND greater than or
 *    equal to the threshold score.
 * 6. If both conditions are met, increment a counter.
 * 7. After checking all participants, print the final count.
 *
 * @time_complexity O(N), where N is the number of participants.
 * @space_complexity O(N) to store the scores.
 */
#include <iostream>

using namespace std;

int main() {
  int n; // Total number of participants
  int k; // The rank that determines the score threshold
  int count = 0; // Counter for participants who advance
  cin >> n >> k;
  int c[n]; // Array to store the scores

  // Read all participant scores.
  for (int i = 0; i < n; i++) {
    cin >> c[i];
  }

  /**
   * @block_logic Loop through each participant to check if they advance.
   * @pre The scores are in array `c`, and `k` is the rank of the threshold score.
   * @post `count` holds the number of participants who meet the criteria.
   */
  for (int i = 0; i < n; i++) {
    // The inner while loop is a convoluted way to check the two conditions.
    // It checks if the current participant's score is positive and
    // greater than or equal to the k-th participant's score.
    while (c[i] > 0) { // Condition 1: Score must be positive.
      if (c[i] >= c[k - 1]) { // Condition 2: Score must be >= k-th score.
        count++;
        break; // Exit the inner while loop after counting.
      }
      break; // Exit if score is not >= k-th score.
    }
  }

  cout << count;
  return 0;
}