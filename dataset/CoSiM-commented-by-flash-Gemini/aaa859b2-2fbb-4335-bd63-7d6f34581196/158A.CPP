/**
 * @file 158A.CPP
 * @brief Logic for qualification counting in competitive environments with non-increasing scores.
 * 
 * Logic: Calculates the number of participants qualifying for the next stage. 
 * Progression requires a score greater than or equal to the k-th finisher's 
 * score, and the score must be strictly positive.
 * 
 * Algorithm: Single-pass scan with threshold-based filtering.
 * Time Complexity: $O(N)$ where N is the number of participants.
 * Space Complexity: $O(1)$ auxiliary space beyond the fixed-size score buffer.
 */

#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n;
  int k;
  // Static buffer for scores (problem constraints usually imply N <= 50).
  int pt[50];
  int count = 0;

  // Functional Utility: Reads total participants (n) and the target rank (k).
  scanf("%d %d", &n, &k);

  // Block Logic: Populates the score array.
  for (int i = 0; i < n; i++)
    scanf("%d", &pt[i]);

  /**
   * Block Logic: Evaluates each participant for qualification.
   * Invariant: pt[] is sorted in non-increasing order.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Logic: Special case handling for zero-score start.
     * If the highest score is zero, no one qualifies (given non-increasing scores).
     */
    if (pt[0] == 0) {
      count = 0;
      break; 
    } else if (pt[i] >= pt[k - 1] && pt[i] != 0) {
      /**
       * Functional Utility: Standard qualification check.
       * Condition 1: Score meets or exceeds the benchmark at rank k.
       * Condition 2: Score is strictly positive.
       */
      count++;
    }
  }

  // Output the total number of qualified participants.
  printf("%d\n", count);

  return 0;
}
