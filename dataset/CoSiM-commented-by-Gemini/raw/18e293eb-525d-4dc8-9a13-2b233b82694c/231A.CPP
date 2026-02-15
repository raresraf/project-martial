/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem using the C language.
 * @details This program counts how many problems a team of three will solve,
 * given that a problem is solved only if at least two out of three members
 * are confident in the solution.
 *
 * @section algorithm Algorithm
 * The algorithm is a straightforward counting process:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter, `count`, to 0 to store the number of solvable problems.
 * 3. Loop `n` times, with each iteration representing one problem.
 * 4. In each iteration, read the three members' opinions (`a`, `b`, `c`),
 *    where 1 represents confidence and 0 represents uncertainty.
 * 5. Check if the sum of the three opinions is 2 or greater. This is a concise
 *    way to verify that at least two members are confident.
 * 6. If the condition is true, increment the `count`.
 * 7. After the loop finishes, print the final `count`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of problems, because of the single loop that iterates `n` times.
 * - Space Complexity: O(1), as the memory usage is constant regardless of the input size.
 */
#include <stdio.h>

int main() {
  int n, i, a, b, c;
  int count = 0; // Counter for the number of problems the team will implement.

  // Read the total number of problems.
  scanf("%d", &n);
  
  // Loop through each of the 'n' problems.
  for (i = 0; i < n; i++) {
    // Read the three opinions (0 or 1) for the current problem.
    scanf("%d %d %d", &a, &b, &c);
    
    // Check if at least two members are sure about the solution.
    // The original condition (sum == 2 || sum == 3) is equivalent to sum >= 2.
    if (a + b + c >= 2) {
      count++;
    }
  }
  
  // Print the final count.
  printf("%d", count);

  return 0;
}