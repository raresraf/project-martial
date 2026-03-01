/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 *
 * @problem A team of three programmers decides to solve 'n' problems. For each
 * problem, they will only attempt it if at least two of them are sure about the
 * solution. Given the decisions for each problem (1 if sure, 0 if not), the
 * task is to count how many problems the team will solve.
 *
 * @algorithm The program iterates through each of the 'n' problems. For each
 * problem, it checks the decisions of the three members. If the sum of their
 * decisions (where '1' means sure) is 2 or greater, a counter is incremented.
 * After checking all problems, the total count is printed.
 *
 * @time_complexity O(N), where N is the number of problems, as the code iterates
 * through the problems twice (once for input, once for processing).
 * @space_complexity O(N) to store the inputs for all 'n' problems.
 * Note: This could be optimized to O(1) space by processing each problem's data
 * as it's read, without storing the entire set.
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  // `count` will store the number of problems to be implemented.
  int count = 0;
  cin >> n;

  // Note: The use of a variable-length array (VLA) `a[n][3]` is a C99 feature
  // and not standard C++, but is supported as an extension by some compilers.
  int a[n][3];

  // Input Phase: Read the decisions for all n problems.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }

  // Processing Phase: Iterate through each problem to see if it will be solved.
  for (int i = 0; i < n; i++) {
    // Check if at least two members are sure. The condition below is a
    // convoluted way of checking if the sum of the three inputs is >= 2.
    if (a[i][0] + a[i][1] == 2 || a[i][1] + a[i][2] == 2 ||
        a[i][2] + a[i][0] == 2) {
      count += 1;
    } else {
      // This else block is redundant; the loop would continue anyway.
      continue;
    }
  }

  // Output the final count of problems to be implemented.
  cout << count;
  return 0;
}
