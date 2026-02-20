/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem using C-style I/O.
 *
 * This program determines the number of problems a team will solve. A problem
 * is solved if at least two out of three members agree (represented by 1s).
 * This implementation uses `scanf` and `printf` for input/output and processes
 * each problem's data without storing it.
 *
 * Algorithm:
 * 1. Read the number of problems, n.
 * 2. Initialize a counter 'cnt' to 0.
 * 3. Loop 'n' times:
 *    a. Read the three opinions (a, b, c).
 *    b. Sum the opinions. If the sum is greater than 1 (i.e., 2 or 3),
 *       increment the counter 'cnt'.
 * 4. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream> // Note: iostream is included but not used. stdio.h would be sufficient.
using namespace std;
int main() {
  // n: number of problems.
  // a, b, c: opinions of the three team members.
  // cnt: total count of problems that will be solved.
  int n, a, b, c, cnt = 0;
  
  // Read the number of problems using C-style formatted input.
  scanf("%d", &n);
  
  /**
   * Block Logic: Main processing loop.
   * Iterates 'n' times, decrementing 'n' in each iteration.
   */
  while (n) {
    scanf("%d", &a);
    scanf("%d", &b);
    scanf("%d", &c);
    
    /**
     * Block Logic: Check for consensus.
     * Since inputs are 0 or 1, `a + b + c` gives the number of 'sure' members.
     * The condition `> 1` is a concise way to check if the sum is 2 or 3.
     */
    if (a + b + c > 1)
      cnt++;
      
    n--;
  }
  
  // Print the final count using C-style formatted output.
  printf("%d", cnt);
}
