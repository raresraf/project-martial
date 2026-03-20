/**
 * @file 231A.CPP
 * @brief Solution for Codeforces problem 231A, "Team".
 * @details This program determines the number of problems a team of three friends
 * will implement. The condition for implementation is that at least two of the
 * three friends must be confident about the solution for a given problem.
 *
 * @section Algorithm
 * The problem is solved by iterating through each problem and checking if the
 * implementation condition is met.
 *
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter, `cnt`, to zero. This counter will track the number of
 *    problems to be implemented.
 * 3. Loop `n` times, once for each problem. In each iteration:
 *    a. Read the three opinions (`a`, `b`, `c`), where 1 means confident and 0 means not.
 *    b. Calculate the sum of the three opinions. This sum equals the number of
 *       confident friends.
 *    c. If the sum is greater than 1 (i.e., 2 or 3), increment the counter `cnt`.
 * 4. After the loop, print the final count `cnt`.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), as the program loops once for each of the `n` problems.
 * - Space Complexity: O(1), since the memory usage is constant regardless of the input size.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  // n: the total number of problems.
  // a, b, c: variables to store the opinion of each of the three friends (1 for sure, 0 for not).
  // sum: stores the number of friends who are sure about a problem's solution.
  // cnt: counts the number of problems that will be implemented.
  int n, a, b, c, sum, cnt = 0;

  scanf("%d", &n);

  /**
   * @brief This loop processes each of the `n` problems.
   * Pre-condition: `cnt` is 0, and `n` holds the number of problems to evaluate.
   * Invariant: After each iteration, `cnt` is the number of problems that met
   * the implementation criteria among those already processed.
   */
  while (n--) {
    sum = 0;
    scanf("%d %d %d", &a, &b, &c);
    sum = a + b + c;

    // The condition for implementation: at least two friends are sure.
    if (sum > 1)
      cnt++;
  }

  // Print the total count of problems to be implemented.
  printf("%d\n", cnt);

  return 0;
}
