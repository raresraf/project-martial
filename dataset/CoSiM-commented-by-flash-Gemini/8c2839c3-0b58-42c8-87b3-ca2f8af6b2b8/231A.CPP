/**
 * @8c2839c3-0b58-42c8-87b3-ca2f8af6b2b8/231A.CPP
 * @brief This program determines the number of problems a team will implement
 * in a programming contest. For each problem, three team members express their
 * confidence (1 if confident, 0 if not). A problem is implemented if at least
 * two team members are confident. This is an alternative implementation of the "A. Team" problem
 * using C-style I/O.
 * Algorithm: Iterative counting with conditional sum.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) for storing counters and problem confidence levels.
 */

#include <cstdio> // Required for C-style input/output functions (scanf, printf)
using namespace std;

int main() {
  // @var numProblems: Total number of problems to evaluate.
  // @var total: Counter for problems that will be implemented.
  // @var a, b, c: Confidence levels (1 or 0) of the three team members for a problem.
  int numProblems = 0, total = 0, a = 0, b = 0, c = 0;
  
  // Block Logic: Read the total number of problems from standard input.
  scanf("%d
", &numProblems);
  
  // Block Logic: Iterate through each problem using a while loop.
  // Invariant: 'numProblems' decreases with each iteration until all problems are processed.
  // Pre-condition: 'numProblems' contains the total count of problems.
  while (--numProblems >= 0) {
    // Block Logic: Read confidence levels for the current problem.
    scanf("%d %d %d
", &a, &b, &c);
    // Block Logic: Check if the sum of confidence levels is greater than 1 (i.e., 2 or 3).
    // If true, at least two team members are confident, and the problem is implemented.
    if (a + b + c > 1) {
      ++total; // Increment the count of implemented problems.
    }
  }
  
  // Post-condition: Print the total number of problems that will be implemented.
  printf("%d
", total); // Print the result followed by a newline.
  return 0; // Explicitly return 0 for successful execution.
}
