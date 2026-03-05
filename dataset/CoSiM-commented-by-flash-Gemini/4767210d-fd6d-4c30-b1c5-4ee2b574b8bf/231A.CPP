/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 *
 * This program helps Petya, Vasya, and Tonya decide which problems to implement
 * for a programming contest. They will implement a problem if at least two of
 * them are sure about the solution. The program reads the number of problems
 * and for each problem, the opinions of the three friends, then counts
 * how many problems will be implemented.
 *
 * Algorithm: Simple counting and conditional logic. For each problem, it sums
 * the opinions of the three friends and checks if the sum is 2 or more.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1).
 */
// https://codeforces.com/problemset/problem/231/A
// 231A Team

#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Main function to count implementable problems based on team opinions.
 *
 * Reads the total number of problems 'n'. For each problem, it reads the
 * opinions of three friends (a, b, c) and increments a counter if at least
 * two friends agree (sum of opinions >= 2). Finally, it prints the total count.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare an integer 'n' for the number of problems and 'count' for implementable problems, initialized to 0.
  int n, count = 0;
  // Read the number of problems 'n' from standard input.
  cin >> n;
  /**
   * @brief Block Logic: Processes each problem.
   *
   * Iterates 'n' times, reading the opinions for each problem and updating the 'count'.
   * Pre-condition: 'n' specifies the total number of problems.
   * Invariant: 'count' accumulates the number of problems that meet the implementation criteria.
   */
  while (n--) {
    // Declare integers for the opinions of three friends.
    int a, b, c;
    // Read the opinions for the current problem.
    cin >> a >> b >> c;
    // Pre-condition: 'a', 'b', 'c' are 0 or 1.
    // Block Logic: Checks if at least two friends agree on the solution.
    if (a + b + c >= 2)
      count++; // Increment the counter if two or more friends agree.
  }
  // Print the total number of problems that will be implemented.
  cout << count;
}
