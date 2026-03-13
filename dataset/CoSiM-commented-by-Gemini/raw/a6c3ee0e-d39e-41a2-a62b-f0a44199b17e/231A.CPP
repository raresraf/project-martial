/**
 * @file 231A.CPP
 * @brief Solution for CodeForces Problem 231A - "Team".
 * 
 * @details
 * The problem asks to count the number of problems that will be implemented by a team of three.
 * The team decides to implement a problem if at least two of them are sure about the solution.
 * The input consists of `n` lines, each containing three integers (0 or 1 for each member).
 * '1' means a member is sure, '0' means they are not.
 *
 * @algorithm
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for implementable problems, `count`, to 0.
 * 3. Loop `n` times for each problem:
 *    a. Read the three members' opinions (`a`, `b`, `c`).
 *    b. Sum their opinions.
 *    c. If the sum is 2 or greater, it means at least two members are sure. Increment `count`.
 * 4. Print the final `count`.
 *
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int n;      // Total number of problems.
  int a, b, c; // Opinions of the three team members for a single problem.
  int count = 0; // Counter for problems that will be implemented.
  
  cin >> n;

  /**
   * @brief Main loop to iterate through each problem.
   * Invariant: `count` holds the number of implementable problems from the
   * problems processed so far.
   */
  while (n--) {
    cin >> a >> b >> c;
    
    // If the sum of opinions is 2 or more, the team will solve the problem.
    if (a + b + c >= 2) {
      count++;
    }
  }
  
  // Output the total count of implementable problems.
  cout << count;
  
  return 0;
}
