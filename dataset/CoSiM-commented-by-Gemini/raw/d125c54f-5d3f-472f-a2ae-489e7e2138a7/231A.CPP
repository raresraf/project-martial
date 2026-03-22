/**
 * @file 231A.CPP
 * @brief Counts problems solvable by a team based on member confidence.
 *
 * This program solves the "Team" problem, where a problem is solved if at
 * least two out of three team members are confident. This implementation
 * explicitly counts the number of confident members for each problem.
 *
 * @b Algorithm: Direct Counting
 *   For each problem, the code counts how many of the three inputs are equal
 *   to 1. If this count is two or greater, a master counter for solvable
 *   problems is incremented.
 *
 * @b Time_Complexity: O(N), where N is the number of problems.
 * @b Space_Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // n: the total number of problems to evaluate.
  int n;
  cin >> n;

  // answer: counts the number of problems that will be solved.
  int answer = 0;

  /**
   * @brief Main loop to iterate through each of the 'n' problems.
   */
  for (int i = 0; i < n; i++) {
    int a1, a2, a3;
    cin >> a1 >> a2 >> a3;

    // num_0: declared but unused.
    int num_0 = 0;
    // num_1: counts confident members for the current problem.
    int num_1 = 0;

    // Explicitly count the number of '1's.
    if (a1 == 1) {
      num_1++;
    }
    if (a2 == 1) {
      num_1++;
    }
    if (a3 == 1) {
      num_1++;
    }

    // A problem is solved if 2 or more members are confident.
    if (num_1 >= 2) {
      answer++;
    }
  }

  // Print the total count of solvable problems.
  cout << answer;
}