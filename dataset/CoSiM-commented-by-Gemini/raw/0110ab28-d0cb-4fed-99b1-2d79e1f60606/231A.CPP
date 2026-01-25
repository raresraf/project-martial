/**
 * @file 231A.CPP
 * @brief Counts the number of problems that will be implemented based on team confidence.
 *
 * This program solves a problem where `n` teams are given `n` problems. For each problem,
 * three team members decide if they are sure about the solution (1 for sure, 0 for not sure).
 * A problem will be implemented if at least two members are sure. This program counts
 * the number of problems that will be implemented.
 *
 * @b Algorithm: The program iterates `n` times, once for each problem. In each iteration,
 * it reads the three members' confidence levels and sums them up. If the sum is 2 or greater,
 * it increments a counter for the problems to be implemented.
 *
 * @b Time_Complexity: O(N), where N is the number of problems (and teams), as it processes each
 * problem's data once.
 * @b Space_Complexity: O(1), as it uses a fixed number of variables to store counts and temporary values.
 */
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, count = 0, line_count, i, temp;
  // Read the total number of problems.
  cin >> n;
  /**
   * @brief This loop iterates through each problem to determine if it will be implemented.
   *
   * Pre-condition: `n` is the number of problems.
   * Invariant: `count` holds the number of problems that have met the implementation criteria so far.
   */
  while (n--) {
    line_count = 0;
    // For each problem, read the confidence level of the three team members.
    for (i = 0; i < 3; i++) {
      cin >> temp;
      line_count += temp; // Sum up the confidence levels.
    }
    // Block Logic: If two or more members are confident, the problem is implemented.
    if (line_count >= 2)
      count++;
  }
  // Output the total count of problems to be implemented.
  cout << count << endl;
  return 0;
}
