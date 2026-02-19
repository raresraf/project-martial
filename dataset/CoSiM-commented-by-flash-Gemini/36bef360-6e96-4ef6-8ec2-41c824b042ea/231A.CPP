/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem: "Team".
 *
 * This problem involves determining how many problems a team will implement
 * given the opinions of three friends for 'n' problems. A problem is implemented
 * if at least two of the three friends are sure about the solution.
 *
 * Algorithm: Iterative counting of problems where at least two friends agree.
 * Time Complexity: O(N), where N is the number of problems.
 */
#include <iostream> // Functional Utility: Included for standard input/output operations.
// using namespace std; // Functional Utility: This line is often used to avoid prefixing standard library elements with std::
                     // but is commented out or not present, meaning elements like cin, cout, endl would need std:: prefix.
                     // The problem uses scanf/printf from <cstdio> which is implicitly included by <iostream> in some compilers or often explicitly included.

int main() {
  // Variable Declarations:
  // n: Total number of problems.
  // a, b, c: Opinions of the three friends (1 if sure, 0 if not sure).
  // cnt: Counter for the number of problems that will be implemented.
  int n, a, b, c, cnt = 0;

  // Input: Reads the total number of problems from standard input.
  scanf("%d", &n);

  // Block Logic: Iterates through each problem.
  // Invariant: At the start of each iteration, 'n' represents the remaining problems to process.
  while (n) {
    // Input: Reads the opinions of the three friends for the current problem.
    scanf("%d", &a);
    scanf("%d", &b);
    scanf("%d", &c);

    // Block Logic: Checks if at least two friends are sure about the solution.
    // Precondition: a, b, c are either 0 or 1.
    // The sum a + b + c will be greater than 1 if at least two variables are 1.
    if (a + b + c > 1)
      cnt++; // Inline: Increments the counter if the problem will be implemented.
    n--;     // Inline: Decrements the remaining problem count.
  }

  // Output: Prints the total count of problems that will be implemented.
  printf("%d", cnt);
  return 0; // Functional Utility: Indicates successful program execution.
}
