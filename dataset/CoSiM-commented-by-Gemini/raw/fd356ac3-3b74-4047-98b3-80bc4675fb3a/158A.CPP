/**
 * @file 158A.cpp
 * @brief Solution to the "Next Round" competitive programming problem.
 * This program determines how many participants will advance to the next round of a contest.
 * A participant advances if their score is positive and greater than or equal to the score
 * of the participant who finished in k-th place.
 *
 * The file includes a large template of helper functions and typedefs, which is a common
 * practice in competitive programming for faster coding.
 *
 * Algorithm:
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read the scores of all `n` participants into an array.
 * 3. Handle the edge case where all scores are 0.
 * 4. Iterate through the participants' scores:
 *    a. Count a participant if their score is greater than or equal to the k-th
 *       participant's score and their score is also positive.
 * 5. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N) to store the scores.
 */
#include <cstdio>
#include <iostream>
// ... other includes ...

// The following is a large template of utility functions and type definitions,
// commonly used in competitive programming to have a ready-made set of tools
// for various problems. It includes macros, typedefs for common data structures,
// and helper functions for tasks like checking min/max, string conversion, etc.
// BEGINTEMPLATE_BY_ZAYED
// ... template code ...
// ENDTEMPLATE_BY_ZAYED

int a[110];

int main() {
  int n, K, cnt = 0;
  // Read the number of participants and the k-th rank.
  scanf("%d%d", &n, &K);
  for (int I = 1; I <= n; I++)
    scanf("%d", &a[I]);

  // A slightly complex way to check if all scores are 0.
  int I;
  for (I = 1; I <= n; I++) {
    if (a[I] != 0)
      break;
  }
  if (I - 1 == n) {
    printf("%d\n", cnt);
    return 0;
  }

  // The main logic: count participants who advance.
  // A participant advances if their score is >= the k-th score and is positive.
  for (I = 1; I <= n; I++) {
    if (a[I] >= a[K] && a[I] > 0)
      cnt++;
  }
  printf("%d\n", cnt);
  return 0;
}
