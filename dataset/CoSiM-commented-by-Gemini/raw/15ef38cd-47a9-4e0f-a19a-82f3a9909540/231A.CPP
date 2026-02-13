/**
 * @file 231A.CPP
 * @brief Space-efficient solution for the "Team" problem.
 * @details This program counts the number of problems a team of three will implement.
 * The rule is that a problem is implemented if at least two of the three members
 * are confident in the solution. This implementation uses `scanf` for input and
 * processes each problem's data as it is read, making it O(1) in space.
 *
 * @section algorithm Algorithm
 * 1. Read the number of problems, `T`.
 * 2. Initialize a `counter` for solvable problems to 0.
 * 3. Loop `T` times, once for each problem.
 *    a. Read the three integer decisions (0 or 1) for the current problem.
 *    b. Sum the three decisions into a local variable `sum`.
 *    c. If `sum` is 2 or greater, increment the `counter`.
 *    d. The line `sum = 0;` at the end of the loop is redundant, as `sum` is a
 *       local variable that goes out of scope and is re-initialized in the next iteration.
 * 4. After the loop, print the final `counter` value using `printf`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(T), where T is the number of problems.
 * - Space Complexity: O(1), as data for each problem is processed on the fly.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int T, counter = 0;
  scanf("%d", &T);
  // Loop T times, once for each problem.
  while (T--) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    // Sum the decisions for the current problem.
    int sum = a + b + c;
    // If two or more members agree, increment the counter.
    if (sum >= 2)
      counter++;
    // This assignment is redundant because `sum` is a local variable to the
    // `while` loop's block and will be out of scope before the next iteration.
    sum = 0;
  }
  printf("%d\n", counter);

  return 0;
}
