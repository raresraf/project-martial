/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @note The original file included a link to the problem description:
 *       https://codeforces.com/problemset/problem/231/A
 *
 * This program calculates the number of problems a three-person team will solve.
 * The condition for solving a problem is that at least two of the three members
 * are confident in the solution.
 *
 * @algorithm
 * 1. Read an integer `n`, which is the total number of problems.
 * 2. Initialize a result counter `r` to 0.
 * 3. Loop `n` times, with each iteration representing one problem.
 *    a. Read the three members' confidence levels (`a`, `b`, `c`), where 1 means
 *       confident and 0 means not confident.
 *    b. Sum the three values.
 *    c. If the sum is 2 or greater, it means at least two members are confident.
 *       Increment the result counter `r`.
 * 4. After the loop completes, print the final count `r`.
 *
 * @complexity
 * Time Complexity: O(n), as the program iterates `n` times, performing a constant
 * amount of work in each iteration.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: total number of problems
  // a, b, c: confidence levels of the three members
  // r: the result counter for problems to be solved
  int n, a, b, c, r;
  cin >> n;
  
  // Initialize the result counter to zero.
  r = 0;
  
  /**
   * @brief Main loop to process each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    // Read the confidence levels for the current problem.
    cin >> a >> b >> c;
    
    // Check if the number of confident members is at least 2.
    if (a + b + c >= 2)
      // If the condition is met, increment the result counter.
      ++r;
  }
  
  // Print the total number of problems the team will solve.
  cout << r << endl;
  
  return 0;
}
