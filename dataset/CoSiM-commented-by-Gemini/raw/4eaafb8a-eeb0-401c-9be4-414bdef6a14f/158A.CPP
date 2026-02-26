/**
 * @file 158A.CPP
 * @brief A clear solution to the "Next Round" problem from Codeforces.
 * @details This program counts participants advancing to the next round. A
 * participant advances if their score is positive and greater than or equal to
 * the score of the k-th ranked participant.
 *
 * @algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array.
 * 3. Identify the threshold score by accessing the score at the k-1 index of the
 *    (non-increasingly sorted) input array. Store this score in a variable.
 * 4. Iterate through the array of scores again.
 * 5. For each score, check if it's greater than or equal to the threshold score
 *    and also positive.
 * 6. Increment a counter for each score that meets these conditions.
 * 7. Print the final counter.
 *
 * @complexity
 * Time Complexity: O(N), for two passes over the `N` participant scores.
 * Space Complexity: O(N), to store the scores in an array.
 */
#include <iostream>
using namespace std;

int main() {
  int arr[51];    // Array to store participant scores.
  int n, k;       // n: number of participants, k: the rank for the threshold.
  int result = 0; // Counter for participants who advance.
  
  cin >> n >> k;

  /**
   * @block
   * @brief Read all participant scores into the array.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  // Adjust k to be a zero-based index.
  k -= 1;
  // Store the score of the k-th participant as the threshold value.
  int threshold_score = arr[k];

  /**
   * @block
   * @brief Count the number of participants who advance.
   */
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is >= the threshold and is positive.
    if (arr[i] >= threshold_score && arr[i] > 0) {
      result++;
    }
  }

  // Print the final count.
  cout << result;
  return 0;
}