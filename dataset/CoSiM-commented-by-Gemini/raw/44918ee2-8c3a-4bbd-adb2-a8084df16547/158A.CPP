/**
 * @file raw/44918ee2-8c3a-4bbd-adb2-a8084df16547/158A.CPP
 * @brief Solution to Codeforces problem 158A - Next Round.
 *
 * @algorithm
 * The problem is to determine how many participants will advance to the next
 * round. The rule is that a participant advances if their score is equal to
 * or greater than the k-th place finisher's score, provided their score is
 * also positive. The scores are given in non-increasing order.
 *
 * The algorithm implemented is as follows:
 * 1. Read the total number of participants, `n`, and the rank `k`.
 * 2. Dynamically allocate an array to store the `n` scores.
 * 3. Read all `n` scores from standard input into the array.
 * 4. Identify the threshold score. Since the scores are in non-increasing order
 *    and 1-based rank `k` corresponds to 0-based index `k-1`, the threshold
 *    score is `score[k-1]`.
 * 5. Initialize a counter `count` to 0.
 * 6. Iterate through all the scores from the beginning of the array.
 * 7. For each score, check if it meets two conditions:
 *    a) The score is greater than or equal to the threshold score.
 *    b) The score is positive (greater than 0).
 * 8. If both conditions are true, increment the `count`.
 * 9. After checking all scores, print the final `count`.
 *
 * @complexity
 * - Time Complexity: O(n), for reading the scores and then iterating through
 *   them once to count the qualifiers.
 * - Space Complexity: O(n), due to the dynamically allocated array to store
 *   the scores of the `n` participants.
 */
#include <iostream>
#include <new> // Included for `new`, though iostream is often sufficient.
using namespace std;

int main() {
  // n: total number of participants.
  // k: the rank of the participant whose score is the threshold.
  // count: number of participants who advance to the next round.
  // thresh: the score of the k-th participant.
  int n, k, count = 0, thresh;
  int *score;

  // Read number of participants and the k-th rank.
  cin >> n >> k;

  // Dynamically allocate memory for the scores.
  score = new int[n];

  /**
   * @brief Read all participant scores into the array.
   */
  for (int i = 0; i < n; i++) {
    cin >> score[i];
  }

  // The threshold score is the score of the k-th participant (at index k-1).
  thresh = score[k - 1];

  /**
   * @brief Count the number of participants who advance.
   * A participant advances if their score is >= the threshold and is positive.
   */
  for (int j = 0; j < n; j++) {
    if (score[j] >= thresh && score[j] > 0)
      count++;
  }

  // Output the final count.
  cout << count;

  // The dynamically allocated memory is not freed with `delete[]`.
  // This is common in short-lived competitive programming solutions
  // where memory is reclaimed by the OS upon program termination.
  return 0;
}
