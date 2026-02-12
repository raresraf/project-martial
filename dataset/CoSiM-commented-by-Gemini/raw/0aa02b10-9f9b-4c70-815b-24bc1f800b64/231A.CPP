/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem using C-style I/O.
 * @details This program determines the number of problems a three-person team will solve.
 * It reads an integer `n` (the number of problems), then loops `n` times. In each loop,
 * it reads the binary opinions (0 or 1) of the three members. A problem is solved if at least two members
 * agree (i.e., their opinion is 1). The program counts and prints the total number of problems that will be solved.
 * This version uses `scanf` and `printf` for input/output.
 *
 * @algorithm Counting based on pairwise agreement.
 * @complexity
 * Time complexity: O(N), where N is the number of problems, as it iterates through each problem once.
 * Space complexity: O(1), as it only uses a few variables for the calculation.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of problems.
  // cn: a counter for the number of problems that will be solved.
  int n, cn = 0;
  scanf("%d", &n);
  int i;
  /**
   * @brief Main loop to iterate through each problem.
   * @invariant This loop runs `n` times to process each problem.
   */
  for (i = 0; i < n; i++) {
    // a, b, c: the opinions of the three team members.
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    /**
     * @brief Checks if at least two members agree.
     * @pre The opinions `a`, `b`, and `c` have been read.
     * @post The counter `cn` is incremented if any pair of members agrees.
     * @note This logic is equivalent to checking if the sum of a, b, and c is 2 or more.
     */
    if (a == 1 && b == 1)
      cn++;
    else if (b == 1 && c == 1)
      cn++;
    else if (c == 1 && a == 1)
      cn++;
  }
  // Print the final count of solvable problems.
  printf("%d\n", cn);
  return 0;
}