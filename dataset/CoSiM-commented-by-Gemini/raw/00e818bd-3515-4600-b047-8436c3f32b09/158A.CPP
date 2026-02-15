/**
 * @file 158A.CPP
 * @brief Determines the number of contestants advancing to the next round.
 *
 * This program calculates the number of contestants who will advance based on a specific rule:
 * a contestant advances if they achieve a positive score and their score is at least as high
 * as the k-th place contestant's score.
 *
 * @b Algorithm: The program reads `n` (number of contestants) and `k`. It initializes the number
 * of passing contestants to `k`. It then iterates through the first `k` scores, decrementing
 * the pass count for any contestant with a score of 0 and establishing the k-th score as the
 * threshold. For the remaining `n-k` contestants, it increments the pass count if their score
 * is positive and meets or exceeds the threshold.
 *
 * @b Time_Complexity: O(N), where N is the number of contestants, due to the single pass.
 * @b Space_Complexity: O(1), as it uses a fixed amount of storage.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  // Read the total number of contestants and the rank k.
  cin >> n >> k;

  int pass = k;
  int base = -1;

  /**
   * @brief This loop iterates through all contestant scores to adjust the final count of
   * those who advance.
   *
   * Pre-condition: `n` and `k` are initialized. `pass` is optimistically set to `k`.
   * Invariant: `pass` is adjusted based on the scores encountered so far. `base` will
   * hold the score of the k-th contestant.
   */
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;

    // Block Logic: This section processes the first k contestants.
    if (i <= k - 1) {
      // Sets the score of the k-th contestant as the threshold.
      if (i == k - 1)
        base = temp;
      // Contestants in the top k with a score of 0 do not advance.
      if (temp == 0) {
        pass--;
      }
    } else { // This else block was missing in the original search string, causing a mismatch.
      // Block Logic: This section processes contestants after the k-th position.
      // An extra contestant passes if their score is positive and meets or exceeds the threshold.
      if (temp > 0 && temp >= base)
        pass++;
    }
  }

  cout << pass << endl;

  return 0;
}