/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem in C.
 * @details This program calculates the number of problems a team will solve.
 * The condition for solving a problem is that at least two out of three team
 * members are sure about the solution (represented by 1).
 *
 * @section algorithm Algorithm
 * 1. Read an integer `n`, which is the total number of problems.
 * 2. Initialize a counter `count` to 0. This will store the number of solved problems.
 * 3. Loop from `i = 0` to `n-1` to process each problem.
 *    a. Inside the loop, read the three integer opinions (`a`, `b`, `c`) for the current problem.
 *    b. Calculate the sum of the three opinions.
 *    c. If the sum is equal to 2 or 3, it means at least two members are confident.
 *       In this case, increment the `count`.
 * 4. After the loop finishes, print the final value of `count`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), because the program iterates once for each of the `n` problems.
 * - Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */
#include <stdio.h>

/**
 * @brief Main function to execute the problem-solving logic.
 * @return 0 on successful execution.
 */
int main() {
  int n, i, a, b, c, count = 0;
  
  // Read the total number of problems.
  scanf("%d", &n);
  
  // Loop through each problem.
  for (i = 0; i < n; i++) {
    // Read the opinions of the three members.
    scanf("%d %d %d", &a, &b, &c);
    
    // Check if the sum of opinions is 2 or 3 (i.e., at least 2).
    if (a + b + c >= 2) {
      count++;
    }
  }
  
  // Print the total count of solvable problems.
  printf("%d", count);

  return 0;
}