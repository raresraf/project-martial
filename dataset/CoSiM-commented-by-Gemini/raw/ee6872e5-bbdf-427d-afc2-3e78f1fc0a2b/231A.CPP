/**
 * @file 231A.CPP
 * @brief Counts the number of problems to be implemented based on team confidence.
 *
 * This program reads the number of problems and for each problem, it takes three
 * boolean inputs representing the confidence of three team members. It counts
 * how many problems have at least two members confident about the solution.
 *
 * @b Algorithm: The program iterates through each problem, sums up the confidence
 *    votes (0 or 1) for the three members, and if the sum is 2 or greater, it
 *    increments a counter for the problems to be implemented.
 * @b Time_Complexity: O(N), where N is the number of problems, as it processes
 *    each problem in a single pass.
 * @b Space_Complexity: O(1), as it only uses a few variables to store the counts
 *    and inputs for each problem.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n; // Number of problems.
  cin >> n;
  int ans = 0; // Counter for problems to be implemented.
  
  /**
   * @brief Loop to process each problem.
   * Reads the confidence levels for three members (a, b, c) for each problem.
   * If the sum of their confidences is greater than 1 (i.e., at least two are sure),
   * the problem is counted.
   */
  while (n != 0) {
    int a, b, c; // Confidence of the three members (1 for sure, 0 for not sure).
    cin >> a >> b >> c;
    if (a + b + c > 1)
      ans += 1;
    n--;
  }
  
  cout << ans << endl;
  return 0;
}