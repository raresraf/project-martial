/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A - Next Round.
 *
 * @algorithm
 * The problem is to count how many participants advance to the next round. The
 * rule is that a score must be positive and >= the k-th participant's score.
 * The input scores are in non-increasing order.
 *
 * This solution is highly memory-efficient as it does not store the scores in
 * an array. It reads them sequentially and determines the cutoff point.
 *
 * The algorithm is as follows:
 * 1. Read `n` and `k`.
 * 2. Loop `k` times, reading the first `k` scores. The score of the k-th
 *    participant is stored in `a`.
 * 3. During this first loop, if any participant's score is 0, it means no one
 *    from this point on can qualify (due to the non-increasing order and the
 *    "must be positive" rule). The number of qualifiers is the current index `i`,
 *    so we print `i` and exit.
 * 4. If the first loop completes, all top `k` scores were positive. The variable `a`
 *    now holds the score to beat.
 * 5. A second loop reads the remaining `n-k` scores.
 * 6. In this loop, if a participant's score `b` is not equal to `a`, it must be
 *    less than `a` (due to non-increasing order). This is the first person who
 *    doesn't qualify. The number of qualifiers is the current index `i`, so we
 *    print `i` and exit.
 * 7. If both loops complete without exiting, it means all `n` participants
 *    qualified. Print `n`.
 *
 * @complexity
 * - Time Complexity: O(N), for the single pass reading the N scores.
 * - Space Complexity: O(1), as no array/vector is used to store scores.
 */
#include <stdio.h>

int main() {
  int n, k;

  scanf("%d%d", &n, &k);

  int a; // Will hold the score of the k-th participant.

  /**
   * @brief Read the first k scores to find the cutoff score and check for zeros.
   */
  for (int i = 0; i < k; i++) {
    scanf("%d", &a);
    // If any of the top k scores is 0, no one from this point on qualifies.
    // The number of qualified participants is `i`.
    if (a == 0) {
      printf("%d", i);
      return 0;
    }
  }

  int b; // Score of participants after the k-th one.
  /**
   * @brief Read the remaining n-k scores to find the first non-qualifier.
   */
  for (int i = k; i < n; i++) {
    scanf("%d", &b);
    // If a score `b` is not equal to the k-th score `a`, it must be less.
    // This is the first person who fails to qualify. The total number
    // of qualified people is `i`.
    if (b != a) {
      printf("%d", i);
      return 0;
    }
  }

  // If the program reaches here, it means all `n` participants qualified.
  printf("%d", n);
  return 0;
}
