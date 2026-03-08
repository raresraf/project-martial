/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem using C-style I/O.
 *
 * @details
 * This program counts how many problems a team of three will implement, based
 * on the condition that at least two members must be confident. It reads the
 * number of problems and then iterates, summing the confidence votes for each
 * problem to see if the threshold of 2 is met. This version uses `scanf` and
 * `printf` for input/output.
 *
 * @section
 * Algorithm: Iterative counting using a summation condition.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <cstdio>
using namespace std;

int main() {
  // numProblems: the total number of problems to evaluate.
  // total: the accumulator for the number of problems that will be solved.
  // a, b, c: confidence votes (0 or 1) for a single problem.
  int numProblems = 0, total = 0, a = 0, b = 0, c = 0;
  scanf("%d
", &numProblems);
  /**
   * @brief This loop iterates through each of the 'numProblems'.
   *
   * Logic: It uses a pre-decrement while loop, which is a compact way to
   * iterate a fixed number of times.
   */
  while (--numProblems >= 0) {
    scanf("%d %d %d
", &a, &b, &c);
    /**
     * @brief Check if the team will implement the problem.
     *
     * Logic: Summing the three votes and checking if the result is greater
     * than 1 is an efficient way to determine if at least two members are sure.
     */
    if (a + b + c > 1) {
      ++total;
    }
  }
  // Print the final count of solvable problems.
  printf("%d
", total);
  return 0;
}
