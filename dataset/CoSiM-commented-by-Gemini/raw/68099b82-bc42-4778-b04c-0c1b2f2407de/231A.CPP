#include <iostream>
using namespace std;

/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem from Codeforces (Problem 231A).
 * @details This program calculates how many problems a team of three will solve.
 * The team solves a problem only if at least two of the three members are confident
 * about the solution.
 *
 * Algorithm:
 * The program first reads the total number of problems, `n`. It then enters a loop
 * that repeats `n` times. In each iteration, it reads the decisions of the three
 * members (represented by 1 for confident, 0 for not confident). It sums these
 * decisions. If the sum is 2 or greater, it means at least two members are confident,
 * so a counter for the number of solved problems is incremented. After the loop
 * finishes, the program prints the final count.
 *
 * Time Complexity: O(n), where `n` is the number of problems, as it processes each
 * problem once.
 * Space Complexity: O(1), as the memory used is constant.
 */
int main() {
  int n, a, b, c, cont = 0;
  // Read the total number of problems.
  cin >> n;
  /**
   * @brief Loop to process each problem.
   */
  while (n > 0) {
    // Read the confidence votes for the current problem.
    cin >> a >> b >> c;
    /**
     * @brief Check if at least two members are confident.
     * The sum of votes will be 2 or 3 if two or three members are confident.
     */
    if (a + b + c > 1)
      cont++;
    n--;
  }
  // Print the total number of problems that will be solved.
  cout << cont;
}
