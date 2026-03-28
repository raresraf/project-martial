/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 *
 * This program calculates the number of problems a team of three will solve.
 * The team decides to solve a problem only if at least two members are sure
 * about the solution.
 *
 * @algorithm The program reads the number of problems, `n`. It then iterates `n`
 * times. In each iteration, it reads the certainty of the three members (1 for
 * sure, 0 for not sure). It sums these values. If the sum is 2 or greater, it
 * means a consensus is reached, and a counter for solved problems is incremented.
 *
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(1).
 */
#include <cstdio>
using namespace std;

int main() {
  int n, a, b, c, x = 0;
  scanf("%d", &n);
  /**
   * @block_logic This loop iterates through each of the `n` problems.
   */
  while (n--) {
    a = 0;
    b = 0;
    c = 0;
    scanf("%d%d%d", &a, &b, &c);
    /**
     * @block_logic Checks if at least two team members are sure about the solution.
     * The sum of their inputs (0 or 1) will be >= 2 if this is the case.
     */
    if (a + b + c >= 2)
      x++;
  }
  // Prints the total number of problems the team will solve.
  printf("%d", x);
}
