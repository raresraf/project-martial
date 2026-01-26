/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem from Codeforces.
 *
 * This problem determines the number of participants who advance to the next
 * round of a competition. A participant advances if their score is strictly
 * positive and is greater than or equal to the score of the k-th place finisher.
 *
 * Problem Link: https://codeforces.com/problemset/problem/158/A
 *
 * Algorithm: Linear scan through participant scores to identify the k-th
 *            place score and then count participants meeting the advancement criteria.
 * Time Complexity: O(N), where N is the total number of participants.
 * Space Complexity: O(1), as only a constant number of variables are used
 *                   regardless of the number of participants.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

int main() {
  int n, k; // n: total participants, k: k-th place to compare against.
  // Reads the total number of participants (n) and the k-th place (k).
  cin >> n >> k;

  int pass = 0;   // Counter for participants who advance to the next round.
                  // Initialized to 0, it will be incremented for each valid participant.
  int base = -1;  // Stores the score of the k-th place finisher.
                  // Initialized to -1 to indicate it hasn't been set yet.

  /**
   * Block Logic: Iterates through all participant scores to determine advancement.
   * Functional Utility: This loop identifies the k-th place score and then
   *                     counts participants whose scores meet the criteria for advancement.
   *
   * Invariant: `base` will hold the k-th participant's score after `i == k-1`.
   *            `pass` accumulates the count of advancing participants based on conditions.
   *
   * @note The logic for `pass` is structured to first identify the k-th score
   *       and then apply the full rule, which can be interpreted differently.
   *       The code, as written, appears to directly count valid scores from the beginning.
   */
  for (int i = 0; i < n; i++) {
    int temp; // Current participant's score.
    cin >> temp;

    // Special handling for the first k participants to determine the `base` score.
    if (i == k - 1) {
      base = temp; // Set `base` to the score of the k-th place finisher.
    }

    // Advancement criteria:
    // 1. Score must be strictly positive (temp > 0).
    // 2. Score must be greater than or equal to the k-th place finisher's score (temp >= base).
    // This condition applies to all participants, once `base` is determined.
    if (temp > 0 && temp >= base) {
      pass++; // Increment the counter if the participant advances.
    }
  }

  // Prints the total number of participants who advance to the next round.
  cout << pass << endl;

  return 0;
}
