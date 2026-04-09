/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem from Codeforces, determining
 * how many problems a team of three friends will implement. A problem is
 * implemented if at least two of the three friends are sure about the solution.
 *
 * Algorithm: Iterative problem evaluation with pairwise confidence checks.
 * 1. Read the total number of problems `n`.
 * 2. Initialize a counter `count` to 0 for implemented problems.
 * 3. For each problem:
 *    a. Read three integers (`a`, `b`, `c`) representing each friend's confidence (0 or 1).
 *    b. Check if any two friends (a,b), (a,c), or (b,c) are both confident.
 *    c. If at least two friends are confident, increment `count`.
 * 4. Print the final `count`.
 * Time Complexity: O(N) where N is the number of problems, as each problem is processed in constant time.
 * Space Complexity: O(1) for storing variables.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function of the program.
 * Reads problem data and calculates the number of problems the team will implement.
 * @pre Input starts with an integer `n`, followed by `n` lines, each containing three integers (0 or 1).
 * @post Prints the total count of problems the team will implement.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: 'n' for the total number of problems.
  int n;
  // Functional Utility: 'count' tracks the number of problems the team will implement.
  int count;
  // Functional Utility: Reads the total number of problems.
  cin >> n;
  // Functional Utility: Initializes the count of implemented problems to 0.
  count = 0;

  // Block Logic: Iterates through each problem to determine if it will be implemented.
  // Invariant: `i` is the current problem being evaluated. `count` accurately reflects
  //            the number of problems implemented from those processed so far.
  for (int i = 0; i < n; i++) {
    // Functional Utility: 'a', 'b', 'c' represent the confidence (0 or 1) of each of the three friends.
    int a, b, c;
    // Functional Utility: Reads the confidence levels for the current problem.
    cin >> a >> b >> c;
    // Block Logic: Checks if at least two friends are sure about the solution.
    // This is done by checking all possible pairs.
    // Pre-condition: `a`, `b`, `c` are integers (0 or 1).
    if ((a == 1 && b == 1)) {
      count = count + 1; // Functional Utility: Increments the count if Petya and Vasya are sure.
    } else if ((a == 1 && c == 1)) {
      count = count + 1; // Functional Utility: Increments the count if Petya and Tonya are sure.
    } else if ((b == 1 && c == 1)) {
      count = count + 1; // Functional Utility: Increments the count if Vasya and Tonya are sure.
    }
  }
  // Functional Utility: Prints the final total count of problems the team will implement.
  cout << count << "
";
  return 0;
}
