/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * This program determines the number of problems a team of three friends will implement
 * in a programming contest. A problem is implemented if at least two friends are
 * sure about its solution.
 * Algorithm: Iterates through problems, sums certainty votes, and counts problems with > 1 vote.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream).

using namespace std; // Uses the standard namespace.

#define ll long long // Defines 'll' as a shorthand for 'long long', though not used for problem variables.

int main() {
  int n; // Declares an integer to store the total number of problems.
  cin >> n; // Reads the total number of problems from standard input.
  
  int res = 0; // Initializes a counter for problems that will be implemented.
  int t;     // Temporary variable to read certainty votes.
  
  /**
   * Block Logic: Processes each problem one by one to determine if it will be implemented.
   * The outer loop iterates through each of the 'n' problems.
   * Invariant: `res` holds the number of problems implemented so far.
   */
  for (int i = 0; i < n; i++) { // Outer loop for each problem.
    int a = 0; // Initializes a variable to sum certainty votes for the current problem.
    
    /**
     * Block Logic: Reads the certainty votes of the three friends for the current problem.
     * The inner loop iterates three times, once for each friend.
     * Invariant: `a` accumulates the sum of certainty votes for the current problem.
     */
    for (int j = 0; j < 3; j++) { // Inner loop for the three friends.
      cin >> t; // Reads a friend's certainty vote (1 if sure, 0 if not).
      a += t;   // Adds the vote to the current problem's sum.
    }
    
    // Checks if the sum of certainties is greater than 1 (meaning 2 or 3 friends are sure).
    if (a > 1)
      res++; // If so, increment the counter for implemented problems.
  }
  
  cout << res << "\n"; // Prints the total number of problems that will be implemented.
  
  return 0; // Indicates successful program execution.
}
