
#include <iostream>

using namespace std;

/**
 * @file 158A.CPP
 * @brief An attempted solution to Codeforces problem 158A, "Next Round".
 * @details This program attempts to count the number of participants who qualify
 * for the next round.
 *
 * Algorithm:
 * The code iterates through the list of scores, which are assumed to be in
 * non-increasing order. It keeps track of the current rank (`pos`). The intent
 * is to count all players with a positive score, but to stop counting as soon
 * as the rank of a player drops below the `k`-th rank.
 *
 * Implementation Note:
 * This logic is flawed. The problem requires counting all players whose score
 * is >= the k-th player's score (and positive). This code stops prematurely
 * when the rank changes past `k`, failing to account for players who are tied
 * with the k-th player but have a rank > k. For example, for input k=5 and
 * scores {10, 9, 8, 7, 5, 5, 5}, the correct answer is 7, but this code
 * would output 5.
 *
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N) for the variable-length array storing scores.
 */
int main() {
  int n, k;
  cin >> n >> k;
  // Using a variable-length array (VLA), a non-standard C++ extension.
  int arr[n];
  for (int i = 0; i < n; i++)
    cin >> arr[i];

  int count = 0;
  // Only proceed if the top score is positive.
  if (arr[0] > 0) {
    int prev = arr[0]; // Tracks the previous score to detect rank changes.
    int i = 0;
    int pos = 1; // Tracks the current rank.

    /**
     * @brief Loop to count advancing players based on a flawed rank-check logic.
     */
    while (i < n) {
      // Stop if scores are no longer positive.
      if (arr[i] <= 0)
        break;
      
      // Check if the score has dropped, indicating a change in rank.
      if (arr[i] < prev) {
        prev = arr[i];
        pos = i + 1;
        // Flawed condition: breaks as soon as the rank number exceeds k,
        // ignoring ties.
        if (pos > k)
          break;
      }
      count++;
      i++;
    }
  }
  cout << count;
  return 0;
}
