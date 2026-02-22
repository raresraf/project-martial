/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem (Codeforces 158A).
 *
 * @algorithm
 * The problem asks to count the number of participants who will advance to the
 * next round. A participant advances if their score is positive and also greater
 * than or equal to the score of the k-th place finisher.
 *
 * This solution uses a standard two-pass algorithm:
 * 1. Read the number of participants `N` and the rank `k`.
 * 2. Read all `N` scores into an array `A`.
 * 3. Determine the threshold score by getting the score of the k-th participant
 *    (at index `k-1`).
 * 4. Iterate through the array of scores a second time.
 * 5. For each score, check if it is positive AND greater than or equal to the
 *    threshold score.
 * 6. Count the number of scores that meet both conditions.
 * 7. Print the final count.
 *
 * @complexity
 * - Time Complexity: O(N), for the two separate passes over the N scores.
 * - Space Complexity: O(N), for storing the scores in an array.
 *
 * @note This code is written in a C-style, using `<cstdio>` functions for I/O
 * and a non-standard Variable Length Array (VLA).
 */
#include <cstdio>

using namespace std;

int main() {
  int N, k;
  // Read N and k using C-style formatted input.
  fscanf(stdin, "%d%d", &N, &k);
  
  // A non-standard Variable Length Array (VLA) to store the scores.
  int A[N];
  int i;

  /**
   * @brief First pass: Read all N scores into the array.
   * `A + i` is pointer arithmetic equivalent to `&A[i]`.
   */
  for (i = 0; i < N; ++i) {
    fscanf(stdin, "%d", A + i);
  }
  
  // The threshold score is the score of the k-th participant (at index k-1).
  int num = A[k - 1];
  int count = 0;
  
  /**
   * @brief Second pass: Iterate through the scores to count who advances.
   */
  for (i = 0; i < N; ++i) {
    /**
     * @brief A participant advances if their score is positive AND greater than
     * or equal to the threshold score `num`.
     */
    if (A[i] >= num && A[i] > 0) {
      ++count;
    }
  }
  
  // Print the final count using C-style formatted output.
  fprintf(stdout, "%d", count);

  return 0;
}
