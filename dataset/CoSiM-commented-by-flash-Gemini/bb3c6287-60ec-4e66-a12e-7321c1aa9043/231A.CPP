/**
 * @file 231A.CPP
 * @brief This competitive programming solution addresses a problem where three friends
 *        decide to implement a problem if at least two of them are sure about the solution.
 *        Given a list of problems and each friend's confidence (1 for sure, 0 for not sure),
 *        the task is to count how many problems they will implement.
 * Algorithm: Direct Summation and Conditional Counting.
 *            1. Read the total number of problems `n`.
 *            2. For each problem, read the confidence levels (0 or 1) of the three friends.
 *            3. Sum these three confidence levels. If the sum is 2 or more, it means
 *               at least two friends are sure.
 *            4. Increment a counter for problems that will be implemented.
 *            5. Finally, print the total count.
 * Time Complexity: O(N), where N is the number of problems. The program iterates through
 *                  each problem once, performing a constant number of operations (reading three
 *                  integers, summing them, and comparing).
 * Space Complexity: O(1), as only a few fixed-size integer variables are used.
 */
/*
    Problem Link    :   https://codeforces.com/problemset/problem/231/A
    Contest Link    :   https://codeforces.com/contest/231/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h> // Includes common standard library headers, such as iostream.
using namespace std;

/**
 * @brief Main function of the program. Reads problem data, counts implementable problems, and prints the result.
 * Functional Utility: Serves as the entry point for execution, solving the problem of counting
 *                     problems with sufficient team confidence.
 */
int main() {
  int n; // Declares an integer to store the number of problems.
  cin >> n; // Reads the number of problems.
  int count = 0; // Initializes a counter for problems that will be implemented.

  // Block Logic: Loops `n` times, once for each problem. `n--` decrements `n` after each iteration.
  // Invariant: After each iteration, `count` accurately reflects the number of implementable
  //            problems processed so far.
  while (n--) {
    int x, y, z; // Declares three integers to store the confidence levels of the friends.
    cin >> x >> y >> z; // Reads the confidence levels for the current problem.
    // Block Logic: Checks if the sum of confidence levels is 2 or more.
    // Functional Utility: If `x`, `y`, `z` are 0 or 1, their sum directly indicates
    //                     how many friends are sure about the solution.
    if ((x + y + z) >= 2)
      count++; // If at least two friends are sure, increment the total count.
  }
  cout << count << endl; // Prints the final total count of problems that will be implemented.
  return 0;              // Indicates successful program termination.
}
