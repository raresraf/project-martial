/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A - Next Round.
 *
 * @algorithm
 * The problem is to determine how many participants advance to the next round of a
 * competition. The rule for advancing is that a participant's score must be positive
 * and greater than or equal to the score of the k-th ranked participant. The input
 * scores are provided in non-increasing order.
 *
 * The algorithm is as follows:
 * 1. Read the total number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array.
 * 3. Identify the score of the k-th participant. Since the array is 0-indexed, this
 *    score is at index `k-1`.
 * 4. Iterate through the array of scores again.
 * 5. For each score, check if it is positive (`> 0`) and greater than or equal to the
 *    k-th place score.
 * 6. Increment a counter for each participant who meets these criteria.
 * 7. Print the final count.
 *
 * @note The included headers `<stdio.h>`, `<stdlib.h>`, `<string.h>`, and `<tgmath.h>`
 * are not strictly necessary for this specific implementation which relies on iostream.
 * The array `a` is also fixed-size, which is sufficient for the problem constraints but
 * less flexible than dynamic allocation.
 *
 * @complexity
 * - Time Complexity: O(n), for the two separate loops to read scores and then count qualifiers.
 * - Space Complexity: O(n), to store the `n` participant scores in an array.
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int n, k;

  cin >> n >> k;

  int counter = 0;
  int a[100];
  /**
   * @brief Reads the n scores of the participants into an array.
   * The scores are guaranteed to be in non-increasing order.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  /**
   * @brief Counts the number of participants who advance.
   * A participant advances if their score is positive and is greater than or
   * equal to the score of the k-th place participant (a[k-1]).
   */
  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0)
      counter++;
  }

  // Print the total number of participants who advance.
  cout << counter;
  return 0;
}
