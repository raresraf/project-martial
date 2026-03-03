
#include <algorithm>
#include <cstdio>
using namespace std;

/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A, "Next Round".
 * @details This program counts the number of participants who qualify for the
 * next round based on their scores.
 *
 * Algorithm:
 * A participant qualifies if their score is positive and greater than or equal
 * to the score of the k-th ranked participant. The problem guarantees that the
 * input scores are in non-increasing order.
 *
 * This implementation first reads all scores, then correctly captures the score
 * of the k-th participant (`x[k-1]`) as the passing threshold. It then
 * unnecessarily sorts the array in increasing order. Finally, it iterates
 * through the sorted array and counts how many scores are positive and meet the
 * pre-determined threshold. While the sorting step is redundant and confusing,
 * the logic still yields the correct result because the threshold score is
 * captured before the original ordering is destroyed.
 *
 * Time Complexity: O(N log N) due to the unnecessary sorting operation. A more
 * optimal solution would be O(N).
 * Space Complexity: O(N) to store the scores in an array.
 */
int main() {
  int n, k, pass;
  scanf("%d %d", &n, &k);

  // Use of a variable-length array (VLA), a non-standard C++ extension.
  int x[n];
  int i;
  for (i = 0; i < n; i++) {
    scanf("%d", &x[i]);
  }

  // Capture the score of the k-th participant as the threshold.
  // This is done BEFORE the array is sorted.
  pass = x[k - 1];

  int ans = 0;

  // This sorting step is logically unnecessary given the problem's constraints
  // and the algorithm used, but does not break the final result.
  sort(x, x + n);

  /**
   * @brief Count participants who meet the criteria.
   * This loop iterates through the SORTED array, but compares against the
   * threshold value taken from the UNSORTED array.
   */
  for (i = 0; i < n; i++) {
    if (x[i] != 0 && x[i] >= pass)
      ans++;
  }
  printf("%d", ans);
  return 0;
}
