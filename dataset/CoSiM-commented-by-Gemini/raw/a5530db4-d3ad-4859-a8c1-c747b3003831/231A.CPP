/**
 * @file 231A.CPP
 * @brief Solution for CodeForces Problem 231A - "Team".
 * 
 * @details
 * The problem asks to count the number of problems that will be implemented by a team of three.
 * The team decides to implement a problem if at least two of them are sure about the solution.
 * The input consists of `t` lines, each containing three integers (0 or 1).
 * '1' means a member is sure, '0' means they are not.
 *
 * @algorithm
 * 1. Read the total number of problems, `t`.
 * 2. Initialize a counter for implementable problems, `count`, to 0.
 * 3. Loop `t` times for each problem:
 *    a. Read the three members' opinions (0 or 1).
 *    b. Sum these opinions.
 *    c. If the sum is 2 or greater, it means at least two members are sure. Increment `count`.
 * 4. Print the final `count`.
 *
 * @time_complexity O(T), where T is the number of problems. The inner loop runs a fixed 3 times.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int t; // Total number of problems.
  int c; // Counter for sure members for a single problem.
  int a; // Stores a member's opinion (0 or 1).
  cin >> t;
  int count = 0; // Counter for problems that will be implemented.
  
  /**
   * @brief Main loop to iterate through each problem.
   * Invariant: `count` holds the number of implementable problems from the
   * problems processed so far.
   */
  for (int i = 0; i < t; ++i) {
    c = 0; // Reset sure-member count for the new problem.
    
    /**
     * @brief Inner loop to process the three members' opinions for one problem.
     * Pre-condition: `c` is 0.
     * Post-condition: `c` holds the number of members sure about the current problem.
     */
    for (int j = 0; j < 3; ++j) {
      cin >> a;
      if (a == 1)
        c++;
    }
    
    // If two or more members are sure, the problem gets implemented.
    if (c >= 2)
      count++;
  }

  // Output the total count of implementable problems.
  cout << count;
}
