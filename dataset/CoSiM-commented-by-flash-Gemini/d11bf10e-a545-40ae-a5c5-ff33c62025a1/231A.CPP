/**
 * @file 231A.CPP
 * @brief Partial solution to the "Team" problem (Codeforces 231A).
 *
 * This program is intended to solve a problem where three friends decide whether to implement a problem.
 * A problem is implemented if at least two friends are sure about the solution.
 * The current implementation reads input but does not fully process the logic.
 * Algorithm (Intended): Iterative processing of 'n' problem scenarios, counting scenarios
 *            where at least two out of three confidence values (a, b, c) are positive.
 * Time Complexity (Current): O(N) for reading input, where N is the number of problems.
 * Space Complexity (Current): O(1) for variables, O(N) if 'arr' was used.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function for the "Team" problem.
 *
 * Reads the number of problems and then, for each problem, reads three integers
 * representing friends' confidence. The implementation is currently incomplete.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  // Block Logic: Read the number of problems to be considered.
  // Pre-condition: User provides a single integer 'n'.
  // Invariant: 'n' will hold the count of problems.
  cin >> n;

  // Declared but unused vector. Intended to store results or data, but currently not utilized.
  vector<int> arr;

  // Block Logic: Loop through each problem scenario.
  // Invariant: The loop runs 'n' times, processing one problem's confidence levels per iteration.
  while (n--) {
    int a, b, c;
    // Block Logic: Read three integers representing the confidence levels of three friends
    //              for the current problem.
    // Pre-condition: Three integers are provided for each iteration.
    // Invariant: a, b, and c hold the confidence values for the current problem.
    // Note: The problem logic (e.g., checking if a+b+c >= 2) is missing here.
    cin >> a >> b >> c;
  }
  // Missing: Logic to count problems that meet the criteria and print the final count.
}