/**
 * @file 231A.CPP
 * @brief Counts problems solvable by a team based on members' confidence.
 *
 * This program solves a problem where a team of three decides to solve a set of
 * problems. A problem is tackled only if at least two members are confident.
 * This code reads the confidence levels (1 for confident, 0 otherwise) for 't'
 * problems and counts how many will be solved.
 *
 * @b Algorithm: Direct Counting
 * @b Time_Complexity: O(T), where T is the number of problems.
 * @b Space_Complexity: O(1).
 */
#include <bits/stdc++.h>
// Macro definitions for shorter syntax, common in competitive programming.
#define sc scanf
#define t1(t) scanf("%d", &t)
#define FOR(x) for (int i = 0; i < x; i++)
#define pf printf
#define wh(n) while (n--)
#define ll long long
using namespace std;

int main() {
  // t: the total number of problems to evaluate.
  // count: the number of problems that will be solved.
  int t, count = 0;
  // Read the number of problems using a macro for scanf.
  t1(t);
  
  /**
   * @brief Main loop to iterate through each problem.
   * The `wh(t)` macro is equivalent to `while(t--)`.
   * Invariant: 'count' holds the number of solvable problems from the processed set.
   */
  wh(t) {
    // Read the confidence levels of the three members (x, y, z).
    int x, y, z;
    cin >> x >> y >> z;
    
    /**
     * @brief Check if at least two members are confident.
     * In C++, non-zero integers evaluate to 'true'. This condition checks if any
     * two of the three variables are non-zero.
     */
    if (x && y || x && z || y && z)
      count++;
  }
  
  // Print the total count of solvable problems using a macro for printf.
  pf("%d\n", count);
}
