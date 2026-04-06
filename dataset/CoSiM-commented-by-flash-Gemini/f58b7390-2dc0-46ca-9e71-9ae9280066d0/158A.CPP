/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A - "Next Round".
 * This program calculates the number of participants who will advance to the next
 * round of a competition based on specific scoring criteria.
 *
 * @problem_statement
 * A participant advances to the next round if their score is greater than or
 * equal to the score of the k-th place participant AND their score is strictly positive.
 *
 * @algorithm
 * 1. Read the total number of participants `n` and the k-th place threshold `k`.
 * 2. Read `n` scores into an array.
 * 3. Iterate through the array of scores. For each participant's score:
 *    a. Check if the score is greater than or equal to the score of the k-th place participant (`arr[k-1]`).
 *    b. Check if the score is strictly positive (`arr[i] > 0`).
 *    c. If both conditions are met, increment a counter for advancing participants.
 * 4. Print the final count.
 *
 * Time Complexity: O(N), as the program performs two linear passes over the input array.
 * Space Complexity: O(N), to store the `n` participant scores in an array.
 */
#include <iostream>
using namespace std;
int main() {
  /**
   * @brief Main function to orchestrate the participant advancement calculation.
   * Reads input parameters, processes participant scores, and determines the
   * total count of participants advancing to the next round.
   * @return 0 on successful execution.
   */
  int n; ///< @var n The total number of participants in the competition.
  int k; ///< @var k The 1-based index of the k-th place participant whose score sets a threshold.
  int count = 0; ///< @var count Accumulator for the number of participants who advance.

  // Read the total number of participants and the k-th place threshold.
  cin >> n >> k;

  int arr[n]; ///< @var arr Array to store the scores of all `n` participants.

  /**
   * @brief Loop to read participant scores from standard input.
   * @invariant `i` increments from 0 to `n-1`, populating the `arr` array.
   * @post The `arr` array contains all `n` participant scores.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  /**
   * @brief Loop to evaluate each participant's score against the advancement criteria.
   * @invariant `i` increments from 0 to `n-1`, checking each participant's score.
   * `count` accumulates the number of participants who meet the criteria.
   * @post `count` holds the final number of advancing participants.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief Conditional check for participant advancement.
     * @details A participant advances if their score (`arr[i]`) is:
     * 1. Greater than or equal to the k-th place participant's score (`arr[k-1]`).
     * 2. Strictly positive (not zero or negative).
     */
    if (arr[i] >= arr[k - 1] && arr[i] != 0) {
      ++count; // Increment the count of advancing participants.
    }
  }

  // Output the total number of participants who advance.
  cout << count;
}
