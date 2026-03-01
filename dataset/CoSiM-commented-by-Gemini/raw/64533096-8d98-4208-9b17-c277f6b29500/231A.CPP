/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 *
 * @problem A team of three programmers decides to solve 'n' problems. A problem
 * is solved only if at least two members are sure about the solution. Given the
 * decisions for each problem (1 if sure, 0 if not), count how many problems
 * the team will solve.
 *
 * @algorithm This code uses an optimal O(N) time and O(1) space approach. It
 * processes each problem's data as it's read, without storing the entire input.
 * 1. It loops 'n' times, once for each problem.
 * 2. Inside the loop, it uses a temporary counter (`solved`) for the current problem.
 * 3. An inner loop reads the three members' decisions, incrementing `solved` for
 *    each '1' (a "sure" decision).
 * 4. After reading the three decisions, if `solved` is 2 or more, a total
 *    result counter is incremented.
 * 5. The final result is printed after the main loop finishes.
 *
 * @time_complexity O(N), as it iterates through 'n' problems with a constant
 * number of operations for each.
 * @space_complexity O(1), as input is processed on the fly.
 */
#include <iostream>
#include <vector>
using namespace std;
// Global variables are common in competitive programming but less ideal in larger projects.
int n, e;
int solved;
int result = 0;
vector<int> v; // This vector is declared but never used.

int main() {
  cin >> n;
  // Main loop iterates through each of the 'n' problems.
  for (int i = 0; i < n; i++) {
    // Reset the "sure" vote count for the current problem.
    solved = 0;
    // Inner loop reads the 3 decisions for the current problem.
    // Note: The inner loop variable `i` shadows the outer loop variable `i`.
    for (int i = 0; i < 3; i++) {
      cin >> e;
      if (e == 1) {
        solved++;
      }
    }
    // If at least two members are sure, increment the total result count.
    if (solved >= 2) {
      result++;
    }
  }
  cout << result;
  return 0;
}
