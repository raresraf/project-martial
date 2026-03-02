#include <iostream>
using namespace std;

/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem from Codeforces (Problem 158A).
 * @details This program calculates the number of contestants who will advance to the
 * next round of a competition.
 *
 * Algorithm:
 * The condition for advancing is that a contestant must have a score equal to or
 * greater than the k-th place finisher's score, provided the score is positive.
 * The input consists of `n` participant scores, already sorted in non-increasing order.
 * 1. Read `n` (number of participants) and `k` (the rank to compare against).
 * 2. Read all `n` scores into an array.
 * 3. The score of the k-th place finisher is at index `k-1`. This is the threshold score.
 * 4. Iterate through all `n` scores. Count how many participants have a score that is
 *    greater than or equal to the threshold score AND is also greater than 0.
 * 5. The final count is the number of participants who advance.
 *
 * Time Complexity: O(n), as the program iterates through the scores once to read them
 * and once to check them.
 * Space Complexity: O(n) to store the scores of the participants.
 */
int main() {
  int n, k;
  // Read the number of participants and the rank of the cutoff score.
  cin >> n >> k;

  int scores[n];
  // Read all participant scores.
  for (int i = 0; i < n; i++)
    cin >> scores[i];

  int count = 0;
  /**
   * @brief Count the number of participants who advance.
   * This loop iterates through each participant's score.
   */
  for (int i = 0; i < n; i++)
    /**
     * @brief Check if the participant's score qualifies them for the next round.
     * The score must be positive and greater than or equal to the k-th participant's score.
     * `scores[k-1]` is the cutoff score since the array is 0-indexed and scores are non-increasing.
     */
    if (scores[i] >= scores[k - 1] && scores[i] > 0)
      count++;
  cout << count << endl;

  return 0;
}