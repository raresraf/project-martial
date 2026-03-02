#include <stdio.h>
#include <string.h>

/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem from Codeforces (Problem 158A).
 * @details This program counts the number of contestants who advance to the next round.
 * The rule is that a contestant advances if they have a positive score that is
 * greater than or equal to the score of the k-th place finisher.
 *
 * Algorithm:
 * The program reads the number of contestants `x` and the rank `y`. It then reads `x`
 * scores into an array. It iterates through the array, comparing each contestant's
 * score `c[i]` to the score of the `y`-th ranked contestant, `c[y]`. A counter is
 * incremented for each contestant who has a positive score and a score greater than
 * or equal to the threshold. The code uses 1-based indexing for its array operations.
 *
 * Time Complexity: O(x), where `x` is the number of contestants, due to the loops
 * for reading and checking scores.
 * Space Complexity: O(x) to store the scores in an array.
 */
int main() {
  int x, y, i, cou = 0;
  scanf("%d %d", &x, &y);
  
  // Note: A Variable Length Array (VLA) `c[x]` is used.
  // The loops below use 1-based indexing from 1 to x.
  int c[x]; 

  /**
   * @brief Read `x` scores into the array `c`, using indices 1 through x.
   */
  for (i = 1; i <= x; i++)
    scanf("%d", &c[i]);

  /**
   * @brief Count participants who qualify for the next round.
   */
  for (i = 1; i <= x; i++) {
    // A participant qualifies if their score is positive and >= the y-th person's score.
    if (c[i] > 0 && c[i] >= c[y])
      cou = cou + 1;
  }
  printf("%d", cou);

  return 0;
}
