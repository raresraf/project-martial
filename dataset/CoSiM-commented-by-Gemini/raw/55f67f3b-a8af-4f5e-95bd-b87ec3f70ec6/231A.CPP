/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem is to count how many problems a team of three (Petya,
 * Vasya, and Tonya) will solve. They solve a problem only if at least two of
 * them are sure about the solution.
 *
 * @algorithm
 * The algorithm iterates through each problem and counts the number of members
 * who are confident.
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter `problems` to 0, which will store the final result.
 * 3. Loop `n` times, once for each problem.
 *    a. For each problem, initialize a temporary counter `solve` to 0.
 *    b. Read the confidence levels for Petya, Vasya, and Tonya (1 for sure, 0 for not).
 *    c. For each member, if they are confident (value is 1), increment `solve`.
 *    d. If the `solve` counter for the current problem is 2 or more, increment
 *       the total `problems` counter.
 * 4. After processing all problems, print the final `problems` count.
 *
 * @complexity
 * Time Complexity: O(n), as it makes a single pass through the `n` problems.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  // p, v, t correspond to the confidence of Petya, Vasya, and Tonya.
  int p, v, t;
  // 'solve' is a temporary counter for the current problem.
  int solve;
  // 'problems' is the final count of problems the team will implement.
  int problems;

  cin >> n;

  problems = 0;
  
  /**
   * @block
   * @brief Loop through each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    solve = 0;
    cin >> p >> v >> t;

    // A series of if statements to count the number of confident members.
    // A more compact way would be `solve = p + v + t;`.
    if (p) {
      solve++;
    }
    if (v) {
      solve++;
    }
    if (t) {
      solve++;
    }

    // If two or more members are confident, add it to the total count.
    if (solve >= 2) {
      problems++;
    }
  }
  
  cout << problems;
}
