#include <bits/stdc++.h>
using namespace std;

/**
 * @file 231A.CPP
 * @brief Solution for a competitive programming problem, often called "Team".
 *
 * This program determines the number of problems a team of three will solve.
 * The team decides to solve a problem only if at least two of its three
 * members are confident about the solution.
 *
 * Problem Statement:
 * Given `t` problems, and for each problem, three boolean values (0 or 1)
 * representing the confidence of each of the three team members. Count how
 * many problems the team will implement.
 *
 * Algorithm:
 * 1. Read the total number of problems, `t`.
 * 2. Initialize a counter `x` for solved problems to 0.
 * 3. Loop `t` times for each problem:
 *    a. Read the confidence levels of the three members (`a`, `b`, `c`).
 *    b. Check if the sum of their confidence levels is 2 or greater.
 *    c. If it is, increment the counter `x`.
 * 4. Print the final count `x`.
 *
 * Time Complexity: O(T), where T is the number of problems.
 * Space Complexity: O(1).
 */
int main() {
  long long t, a, b, c, x, i;

  // The 'while' loop suggests handling multiple test case batches,
  // though competitive programming problems usually provide 't' only once.
  while (cin >> t) {
    // Counter for the number of problems to be solved.
    x = 0;
    // Loop through each of the 't' problems.
    for (i = 1; i <= t; i++) {
      cin >> a >> b >> c;

      // Logic: Check if at least two members are confident.
      // A simpler way to write this is: if ((a + b + c) >= 2) { x++; }
      if (a == 1 && b == 1)
        x++;
      else if (a == 1 && c == 1)
        x++;
      else if (b == 1 && c == 1)
        x++;
    }
    // Output the total number of problems the team will solve.
    cout << x << endl;
  }
  return 0;
}