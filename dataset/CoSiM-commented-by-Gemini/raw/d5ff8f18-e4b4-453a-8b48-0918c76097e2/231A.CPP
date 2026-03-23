/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates the number of problems a team of three will implement.
 * The condition for implementation is that at least two of the three members are sure
 * about the solution (represented by 1). This is a straightforward and efficient
 * implementation of the problem's logic.
 *
 * @algorithm Simple Counting
 * @complexity
 * Time: O(N), where N is the number of problems, due to the single loop.
 * Space: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h> // A non-standard header for including all standard libraries.
using namespace std;

int main() {
  int n; // Number of problems
  int a, b, c; // Confidence of the three team members for a single problem
  
  cin >> n;

  int cnt = 0; // Counter for the number of solvable problems

  /**
   * @brief Main loop to process each of the 'n' problems.
   */
  for (int i = 0; i < n; i++) {
    // Read the three confidence values for the current problem.
    cin >> a >> b >> c;

    // Sum the confidence values.
    int sum = a + b + c;

    // If the sum is 2 or more, the team will solve the problem.
    if (sum >= 2) {
      cnt++;
    }
  }

  // Print the final count of solvable problems.
  cout << cnt;

  return 0; // The return is implicit in modern C++ main, but good practice.
}