/**
 * @file 231A.CPP
 * @brief A concise C-style solution in C++ for Codeforces problem 231A, "Team".
 * @details This program counts how many problems a team of three decides to
 * implement, based on the condition that at least two members are confident.
 *
 * @section Algorithm
 * The implementation is a direct and efficient translation of the problem's logic.
 * 1. It reads an integer `n`, the total number of problems.
 * 2. It initializes a `count` of solvable problems to 0.
 * 3. A `for` loop iterates `n` times using a countdown approach.
 * 4. Inside the loop, it reads the three members' opinions (`a`, `b`, `c`),
 *    which are either 1 (confident) or 0 (not confident).
 * 5. It checks if the sum `a + b + c` is greater than 1. This elegantly
 *    verifies if at least two members are confident.
 * 6. If the condition is met, the `count` is incremented.
 * 7. After the loop, the final `count` is printed.
 *
 * This solution is notable for its use of C-style I/O (`scanf`, `printf`) within
 * a C++ file, which is common in competitive programming for performance.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), as it loops once for each of the `n` problems.
 * - Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <cstdio>

using namespace std;

int main() {
  // n: stores the total number of problems.
  int n;
  // count: accumulates the number of problems the team will implement.
  int count = 0;
  scanf("%d", &n);
  /**
   * @brief This loop iterates `n` times using a countdown from `n` to 1.
   * Pre-condition: `count` is 0.
   * Invariant: After each iteration, `count` is the number of implementable
   * problems found in the set of problems processed so far.
   */
  for (int i = n; i > 0; i--) {
    // a, b, c: store the confidence of the three team members.
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    // If the sum of confidences is 2 or 3, the team will implement the problem.
    if (a + b + c > 1) {
      count++;
    }
  }
  // Print the total number of problems to be implemented.
  printf("%d\n", count);
}