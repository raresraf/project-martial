/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A - Next Round.
 *
 * @algorithm
 * The problem is to determine how many contestants will advance to the next round.
 * The rule is: a contestant advances if their score is equal to or greater than
 * the score of the k-th place finisher, and their score is positive.
 *
 * The algorithm implemented is as follows:
 * 1. Read the total number of participants, `n`, and the rank `k` which
 *    determines the score threshold.
 * 2. Read all `n` scores into an array. The scores are given in non-increasing order.
 * 3. Identify the threshold score. This is the score of the k-th participant,
 *    which is at index `k-1` in the 0-indexed array.
 * 4. Iterate through all the scores again.
 * 5. For each score, check if it is greater than or equal to the threshold score
 *    AND it is a positive score.
 * 6. A counter is incremented for each participant who meets these two conditions.
 * 7. Finally, the total count of advancing participants is printed.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of participants. The solution
 *   involves two separate loops that each run up to N times.
 * - Space Complexity: O(N) to store the scores of the participants in an array.
 */
#include <iostream>
#include <new> // Included for `new`, though `<iostream>` often includes it.
using namespace std;

int main() {
  // n: total number of participants.
  // k: the 1-based rank of the participant whose score is the threshold.
  // count: the number of participants who will advance.
  // thresh: the score of the k-th place participant.
  int n, k, count = 0, thresh;
  
  // score: dynamically allocated array to store the scores of participants.
  int *score;

  cin >> n >> k;
  score = new int[n];

  /**
   * Block Logic: Read all participant scores into the array.
   */
  for (int i = 0; i < n; i++) {
    cin >> score[i];
  }

  // The threshold score is the score of the k-th ranked participant.
  // Since the array is 0-indexed, this is at index k-1.
  thresh = score[k - 1];

  /**
   * Block Logic: Count the number of participants who advance.
   * A participant advances if their score is >= the threshold and is positive.
   * Pre-condition: `score` array is filled, `thresh` is set to score[k-1].
   * Invariant: `count` holds the number of advancing participants among the first `j` checked.
   */
  for (int j = 0; j < n; j++) {
    if (score[j] >= thresh && score[j] > 0)
      count++;
  }

  // Output the total count of participants who advance.
  cout << count;
  return 0;
}
