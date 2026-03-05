/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from competitive programming platforms.
 *
 * This program calculates the number of problems a team will decide to implement.
 * The team consists of three members. They will implement a problem only if at least
 * two of them are sure about the solution.
 *
 * @algorithm
 * 1. Read the total number of problems, `t`.
 * 2. Initialize a counter, `ck`, to 0. This counter will store the number of
 *    problems the team will implement.
 * 3. Loop `t` times to process each problem.
 *    a. For each problem, read three integers: `a`, `b`, and `c`. Each integer is
 *       either 1 (sure) or 0 (not sure).
 *    b. Calculate the sum of `a`, `b`, and `c`. This sum represents the number of
 *       members who are sure.
 *    c. If the sum is 2 or greater, it means at least two members are sure. In this
 *       case, increment the counter `ck`.
 * 4. After the loop finishes, print the final value of `ck`.
 *
 * @complexity
 * Time Complexity: O(t), where `t` is the number of problems, as the program
 * processes each problem in constant time inside a loop.
 * Space Complexity: O(1), as a fixed amount of memory is used regardless of the input size.
 */
#include <bits/stdc++.h> // Includes all standard C++ libraries for convenience.
using namespace std;

int main() {
  // `t`: the total number of problems to evaluate.
  int t;
  // `a`, `b`, `c`: represent the confidence of the three team members (1 for sure, 0 for not sure).
  int a, b, c;
  // `ck`: a counter for the number of problems that will be implemented.
  int ck = 0;
  
  // Read the number of problems. `scanf` is used here.
  scanf("%d", &t);
  
  /**
   * @brief This loop iterates through each problem.
   */
  while (t--) {
    // Read the confidence levels for the current problem.
    cin >> a >> b >> c;
    
    // Check if at least two members are sure.
    if ((a + b + c) >= 2) {
      // If so, increment the implementation counter.
      ck++;
    }
  }
  
  // Print the total number of problems to be implemented.
  cout << ck;
  
  return 0;
}
