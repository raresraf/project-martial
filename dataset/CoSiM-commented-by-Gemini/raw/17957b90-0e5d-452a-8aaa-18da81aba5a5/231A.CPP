/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem from a competitive programming context.
 * @details This program calculates the number of problems a team of three will decide to implement.
 * The condition for implementing a problem is that at least two of the three members must be sure about the solution.
 *
 * @section algorithm Algorithm
 * The algorithm is a straightforward counting process:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter, `flag`, to zero. This counter will store the number of problems the team agrees on.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three members' opinions as integers (`a`, `b`, `c`). An opinion is 1 if a member is sure, and 0 otherwise.
 * 5. Calculate the sum of the three opinions. If the sum is 2 or greater, it means at least two members are confident.
 * 6. If the condition is met, increment the `flag` counter.
 * 7. After iterating through all the problems, print the final value of `flag`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of problems, because the program iterates through each problem exactly once.
 * - Space Complexity: O(1), as the program uses a fixed amount of memory regardless of the input size.
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  // Read the total number of problems.
  cin >> n;

  int flag;
  // 'flag' will count the number of problems the team decides to implement.
  flag = 0;

  // Loop through each of the 'n' problems.
  for (int i = 0; i < n; i++) {
    int a, b, c;
    // Read the confidence level of the three team members for the current problem.
    // 1 means sure, 0 means not sure.
    cin >> a >> b >> c;
    // Check if the number of sure members is at least 2.
    if (a + b + c >= 2) {
      // If so, the team will implement this problem.
      flag++;
    }
  }
  // Print the total count of problems to be implemented.
  cout << flag;
  return 0;
}
