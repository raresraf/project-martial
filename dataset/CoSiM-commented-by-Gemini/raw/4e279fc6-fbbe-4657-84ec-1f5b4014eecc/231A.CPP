/**
 * @file 231A.CPP
 * @brief A C-style solution to the "Team" problem from Codeforces.
 * @details This program counts the number of problems a three-person team will
 * solve, where a problem is solved if at least two members are confident. This
 * implementation uses C-style I/O and stores all inputs in arrays before processing.
 *
 * @algorithm
 * 1. Declare three arrays, `P`, `V`, and `T`, to store the confidence levels for
 *    each of the `n` problems.
 * 2. Read the number of problems, `n`.
 * 3. Use a loop to read all `n*3` inputs and populate the `P`, `V`, and `T` arrays.
 * 4. Use a second loop to iterate from `0` to `n-1`. In each iteration, sum the
 *    confidence levels for the i-th problem from the three arrays.
 * 5. If the sum is 2 or greater, increment a result counter.
 * 6. After the second loop, print the final count.
 *
 * @complexity
 * Time Complexity: O(N), for two separate passes over the N problems.
 * Space Complexity: O(N), due to storing all `3*N` inputs in arrays. A more
 * space-efficient O(1) solution is possible by processing each problem's inputs
 * as they are read.
 */

#include <stdio.h>

int main(void) {

  int n;
  int cnt = 0; // Counter for the number of solvable problems.
  
  // Arrays to store the confidence levels of Petya, Vasya, and Tonya for each problem.
  int P[1000];
  int V[1000];
  int T[1000];

  // Read the total number of problems.
  scanf("%d", &n);

  /**
   * @block
   * @brief Read and store all inputs for all `n` problems.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &P[i], &V[i], &T[i]);
  }

  /**
   * @block
   * @brief Process the stored inputs to find the number of solvable problems.
   */
  for (int i = 0; i < n; i++) {
    // Check if the sum of confidence levels for the i-th problem is at least 2.
    if (P[i] + V[i] + T[i] >= 2) {
      cnt++;
    }
  }

  // Print the final count.
  printf("%d", cnt);

  return 0;
}