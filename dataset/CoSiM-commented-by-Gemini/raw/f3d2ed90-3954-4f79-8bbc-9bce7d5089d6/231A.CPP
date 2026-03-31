/**
 * @file 231A.CPP
 * @brief C solution for the "Team" problem using a single-pass sum.
 *
 * This program, written in C, calculates the number of problems a team will
 * solve. It reads the total number of problems, `n`. Then, in a loop, it reads
 * the three members' confidence levels for each problem, calculates their sum,
 * and if the sum is 2 or more, it increments a counter. Finally, it prints the
 * total count of solved problems.
 *
 * @author f3d2ed90-3954-4f79-8bbc-9bce7d5089d6
 *
 * @algorithm Single-Pass Counting
 * @complexity
 * Time: O(N), where N is the number of problems.
 * Space: O(1).
 */
#include <stdio.h>

int main() {
  // `n`: number of problems. `k`: counter for solved problems.
  int n, k = 0;
  scanf("%d", &n);

  /**
   * @brief Main loop to process each of the `n` problems.
   */
  while (n--) {
    // `a`, `b`, `c`: confidence levels of the three members.
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    // `s`: sum of the confidence levels for the current problem.
    int s = a + b + c;
    /**
     * @brief Check if the sum of confidences is 2 or more.
     */
    if (s >= 2)
      k++; // Increment the solved-problem counter.
  }
  // Print the total number of solved problems.
  printf("%d", k);
  return 0;
}