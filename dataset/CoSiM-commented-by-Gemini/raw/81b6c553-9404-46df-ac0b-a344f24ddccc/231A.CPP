/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on member confidence.
 *
 * This program solves the "Team" problem from Codeforces. For a given number of
 * problems, it reads the confidence levels (1 for sure, 0 for not sure) of three
 * team members. It counts how many problems will be implemented, based on the
 * rule that a problem is solved only if at least two members are sure.
 *
 * @algorithm Iterative Counting
 * @complexity
 * Time: O(N), where N is the number of problems, as each problem is processed once.
 * Space: O(1), using a fixed number of variables for the calculation.
 */

#include <iostream>
// The <bits/stdc++.h> header is a non-standard GNU extension often used in
// competitive programming for convenience, as it includes most standard libraries.
// #include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to read problem data and calculate the result.
 */
int main() {
  // n: the total number of problems to evaluate.
  int n;
  cin >> n;
  
  // a, b, c: store the confidence of each team member (1 or 0).
  int a, b, c;
  // co (count): a counter for the number of problems the team will solve.
  int co = 0;
  
  // Pre-condition: 'co' is 0.
  // Block Logic: Loop through each of the 'n' problems.
  while (n--) {
    cin >> a >> b >> c;
    // Check if the sum of confidence levels is 2 or greater.
    if (a + b + c >= 2)
      co++; // Increment the count if at least two members are sure.
  }

  // Output the total number of problems that will be solved.
  cout << co << endl;
  
  // A return statement is good practice, though often omitted in short competitive programs.
  return 0;
}
