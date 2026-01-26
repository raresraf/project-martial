/**
 * @file 231A.CPP
 * @brief Solution to a counting problem, likely from a competitive programming platform.
 * @details This program reads an integer 'n', representing the number of problems. For each
 * problem, it reads three boolean-like integers (0 or 1). It counts how many of these
 * problems have at least two of the three integers equal to 1.
 *
 * @algorithm Simple counting algorithm.
 * @complexity
 * Time complexity: O(n), as it iterates through the 'n' problems once.
 * Space complexity: O(1), as it uses a constant amount of extra space.
 */
#include <cstdio>
using namespace std;

/**
 * @brief Main function to execute the solution.
 */
int main() {
  /**
   * @param n The total number of problems.
   * @param a, b, c Integers representing the three members' solutions for a single problem.
   * @param x Counter for the number of problems that will be implemented.
   */
  int n, a, b, c, x = 0;
  scanf("%d", &n);

  /**
   * @brief Loop 'n' times to process each problem.
   * Pre-condition: 'n' contains the total number of problems to process.
   * Invariant: After each iteration, one problem's inputs have been read and evaluated.
   */
  while (n--) {
    a = 0;
    b = 0;
    c = 0;
    scanf("%d%d%d", &a, &b, &c);

    /**
     * @brief Check if the sum of the three solutions is 2 or more.
     * This is equivalent to checking if at least two members agree.
     */
    if (a + b + c >= 2)
      x++;
  }

  /**
   * @brief Print the final count of problems that meet the condition.
   */
  printf("%d", x);
}
