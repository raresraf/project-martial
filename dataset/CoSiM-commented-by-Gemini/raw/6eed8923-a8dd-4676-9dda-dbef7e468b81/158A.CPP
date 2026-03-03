
#include <stdio.h>

/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A, "Next Round".
 * @details This program counts the number of participants who will advance to
 * the next round of a competition based on their scores.
 *
 * Algorithm:
 * A participant advances if their score is positive and also greater than or
 * equal to the score of the k-th ranked participant. The program first reads
 * the total number of participants (`n`) and the rank to benchmark against (`k`,
 * stored in variable `j`). It reads all scores into an array. It then iterates
 * through the scores a second time, comparing each score to the score of the
 * k-th participant (`a[j-1]`). If a score meets the criteria (>= threshold and
 * > 0), a counter is incremented.
 *
 * Time Complexity: O(N), for reading the N scores and then iterating through
 * them again to count the qualifiers.
 * Space Complexity: O(N) to store the scores in an array.
 */
int main() {
  // a: array to store participant scores.
  int a[100];
  // n: total number of participants.
  int n;
  // j: the rank to use as a benchmark (equivalent to `k` in the problem statement).
  // p: counter for participants who pass.
  int j, p = 0;

  scanf("%d%d", &n, &j);

  // Read all n scores into the array.
  for (int i = 0; i < n; i++) {
    scanf("%d", &a[i]);
  }

  /**
   * @brief Count the number of participants who advance.
   */
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is positive and >= the k-th score.
    if (a[i] >= a[j - 1] && a[i] > 0) {
      p++;
    }
  }

  printf("%d\n", p);

  return 0;
}
