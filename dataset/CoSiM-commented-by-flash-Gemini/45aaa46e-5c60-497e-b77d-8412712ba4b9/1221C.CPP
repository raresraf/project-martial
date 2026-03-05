/**
 * @file 1221C.CPP
 * @brief Solution to a competitive programming problem involving team formation.
 *
 * This program calculates the maximum number of teams that can be formed
 * given a certain number of coaches, mentors, and general participants.
 * Each team requires one coach, one mentor, and one additional person,
 * where general participants can fill either the coach or mentor role if needed.
 *
 * Algorithm: Greedy approach using minimums. The maximum number of teams is
 * limited by the minimum of coaches, mentors, or the total number of people divided by 3.
 * Time Complexity: O(1) per test case, as it involves a few arithmetic operations and min calls.
 * Space Complexity: O(1) per test case.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to process multiple test cases for team formation.
 *
 * For each test case, it reads the number of coaches, mentors, and extra people,
 * then calculates and prints the maximum number of teams that can be formed.
 *
 * @return 0 on successful execution.
 */
int main() {

#ifdef d
  // Pre-condition: Debug mode is enabled, and "input.txt" exists.
  // Block Logic: Redirects standard input from "input.txt" for local testing.
  freopen("input.txt", "r", stdin);
#endif

  // Declare integer 't' for the number of test cases.
  int t;
  // Read the number of test cases.
  cin >> t;
  /**
   * @brief Block Logic: Processes each test case.
   *
   * Invariant: At the start of each iteration, 't' represents the remaining
   * number of test cases to process.
   */
  while (t--) {
    // Declare integers for coaches 'c', mentors 'm', and extra people 'x'.
    int c, m, x;
    // Read 'c', 'm', and 'x' for the current test case.
    cin >> c >> m >> x;
    // Calculate 'y', the minimum of coaches and mentors. This is a primary limiting factor.
    int y = min(c, m);
    // Calculate 'z', the maximum number of teams. This is limited by:
    // 1. 'y' (the minimum of coaches/mentors).
    // 2. The total number of people (c + m + x) divided by 3, as each team needs 3 roles.
    // The minimum of these two values gives the final answer.
    int z = min(y, (c + m + x) / 3);
    // Print the result for the current test case, followed by a newline.
    cout << z << "\n";
  }

  return 0;
}
