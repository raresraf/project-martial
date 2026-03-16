/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem (Codeforces 158A).
 *
 * @details
 * This program determines how many participants will advance to the next round of a contest.
 * The rule is that a participant advances if their score is equal to or greater than the
 * score of the k-th place finisher, as long as their score is positive.
 *
 * The program first reads the total number of participants, `n`, and the rank `k`.
 * It then reads all `n` scores, identifying the score of the k-th participant.
 * Finally, it iterates through all scores again to count how many meet the advancement criteria.
 *
 * @note The use of `int arr[n]` is a Variable Length Array (VLA), which is a
 * non-standard C++ extension, but commonly supported by compilers like g++.
 *
 * Algorithm:
 * 1. Read `n` and `k`.
 * 2. Read all `n` participant scores into an array. While doing so, store the k-th score
 *    as the threshold.
 * 3. Iterate through the array of scores and count how many are greater than or
 *    equal to the threshold and also greater than 0.
 * 4. Print the final count.
 *
 * Time Complexity: O(n), due to two separate linear scans of the participant scores.
 * Space Complexity: O(n), to store the scores of the `n` participants.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int arr[n], count = 0, max;

  /**
   * @brief Reads participant scores and identifies the threshold score.
   * @details This loop populates the `arr` with `n` scores. It specifically
   * captures the score of the k-th participant (at index k-1) to use as the
   * minimum qualifying score for the next round.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    if (i + 1 == k) {
      max = arr[i];
    }
  }

  /**
   * @brief Counts the number of participants who advance to the next round.
   * @details This loop iterates through all participants and checks if their score
   * meets the advancement criteria:
   * 1. The score must be positive.
   * 2. The score must be greater than or equal to the k-th participant's score.
   */
  for (int i = 0; i < n; i++) {
    if (arr[i] >= max && arr[i] > 0) {
      count++;
    }
  }
  cout << count << endl;
  return 0;
}
