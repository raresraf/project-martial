/**
 * @file 231A.CPP
 * @brief Solution for CodeForces Problem 231A, "Team".
 *
 * @description
 * This program determines the number of problems a team of three will implement
 * during a programming contest. The team decides to solve a problem only if at
 * least two of the three members are confident in the solution.
 *
 * @algorithm
 * 1. Read an integer `t`, the total number of problems.
 * 2. Initialize a `counter` to 0 to store the number of problems to be solved.
 * 3. Loop `t` times:
 *    a. For each problem, read the three members' opinions (1 for sure, 0 otherwise).
 *    b. Sum the three opinions.
 *    c. If the sum is 2 or greater, increment the `counter`.
 * 4. After the loop, print the final value of the `counter`.
 *
 * @time_complexity O(T), where T is the number of problems.
 * @space_complexity O(1).
 */
// CodeForces
// Problem 231A [Team]

#include <iostream>
using namespace std;

int main() {
  // x, y, z: represent the opinion of each of the three team members.
  int x, y, z;
  // t: the number of problems.
  // counter: accumulates the count of problems the team will solve.
  int t, counter = 0;
  cin >> t;

  // Block Logic: Iterate through each problem.
  while (t--) {
    cin >> x >> y >> z;
    // Pre-condition: Check if at least two members are sure.
    // Since 'sure' is 1 and 'not sure' is 0, the sum gives the number of
    // sure members.
    if (x + y + z >= 2) {
      counter++;
    }
  }
  cout << counter << endl;
  return 0;
}
