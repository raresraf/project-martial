/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem (Codeforces 158A).
 *
 * Problem Statement:
 * Given the scores of 'n' participants in non-increasing order, determine how many
 * will advance to the next round. A participant advances if they have a positive
 * score and their score is equal to or greater than the k-th place finisher's score.
 *
 * Algorithm:
 * This implementation first reads all 'n' scores into an array 'a'. It then
 * partitions the logic into two main cases based on the score of the k-th
 * finisher (a[k-1]).
 *
 * Case 1: The k-th finisher's score is 0.
 * Since scores are non-increasing and must be positive to advance, the code
 * iterates backward from the k-th position to find the last participant
 * with a score greater than 0. The number of advanced participants is the
 * index of that participant plus one. If all scores up to k are 0, it
 * correctly identifies that 0 participants advance.
 *
 * Case 2: The k-th finisher's score is positive.
 * All participants up to the k-th place are guaranteed to advance. The code then
 * iterates forward from the k-th position to find how many subsequent
 * participants have the same score. It does this by looking for the first
 * drop in score. The number of advanced participants is the index of the last
 * player with the qualifying score, plus one.
 *
 * Note on Implementation:
 * In the second case, the loop `for (int j = k - 1; j < n; j++)` checks `a[j+1]`.
 * When `j` is `n-1`, this results in reading `a[n]`, which is an out-of-bounds
 * access and a potential bug.
 *
 * Time Complexity: O(n), dominated by reading the 'n' scores into the array.
 * Space Complexity: O(n), for storing the 'n' scores in an array.
 */
#include <stdio.h>

int main(void) {

  int n; // The total number of participants.
  int k; // The rank of the participant whose score is the threshold.

  int a[50]; // Array to store the scores of the participants.

  scanf("%d %d", &n, &k);
  for (int i = 0; i < n; i++) {
    scanf("%d ", &a[i]);
  }

  // Case 1: The k-th place finisher has a score of 0.
  // In this case, only people with a positive score can advance.
  if (a[k - 1] == 0) {
    // If the top score is 0, no one advances.
    if (a[0] == 0) {
      printf("%d", 0);
      return 0;
    }
    // Find the last participant before or at k-1 with a non-zero score.
    for (int j = k - 1; j >= 0; j--) {
      if (a[j] != 0) {
        printf("%d", j + 1);
        return 0;
      }
    }
  } 
  // Case 2: The k-th place finisher has a positive score.
  else {
    // Find the first participant after k-1 with a lower score.
    // BUG: This loop reads beyond the array bounds when j = n - 1.
    for (int j = k - 1; j < n; j++) {
      if (a[j] > a[j + 1]) {
        printf("%d", j + 1);
        return 0;
      }
    }
    // If no score drop is found, all 'n' participants advance.
    printf("%d", n);
  }
  return 0;
}
