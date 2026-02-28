/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 231A - "Team").
 *
 * This program calculates the number of problems a team of three friends will implement.
 * The rule for implementation is that at least two of the three friends must be
 * sure about the solution to a given problem.
 *
 * Algorithm: Counting based on a simple conditional sum.
 * The program iterates through 'n' problems. For each problem, it reads three
 * binary inputs (0 or 1), representing each friend's certainty. If the sum of
 * these inputs is 2 or more, a counter is incremented.
 *
 * Time Complexity: O(N), where N is the number of problems, due to a single loop.
 * Space Complexity: O(1), as only a few variables are used irrespective of N.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Configures fast I/O, then processes 'n' problems to count how many will be implemented.
 */
int main() {
  // Functional Utility: Optimize C++ standard streams for faster input/output operations.
  // This is common in competitive programming to meet time limits.
  ios::sync_with_stdio(false);
  cin.tie(0); // Unties cin from cout.

  int n, sum(0), a, b, c;
  // Pre-condition: Read 'n', the total number of problems.
  cin >> n;

  // Block Logic: Iterate 'n' times, once for each problem.
  // Invariant: 'n' decreases with each iteration, loop continues until all problems are processed.
  while (n--) {
    // Pre-condition: Read the opinions of the three friends (a, b, c) for the current problem.
    cin >> a >> b >> c;

    // Block Logic: Check if at least two friends are sure about the solution.
    // If the sum of their opinions (where 1 means sure, 0 means not) is >= 2,
    // then the problem will be implemented.
    if (a + b + c >= 2)
      sum++; // Increment the counter for implemented problems.
  }
  // Output the total number of problems that will be implemented, followed by a newline.
  cout << sum << "\n";
}
