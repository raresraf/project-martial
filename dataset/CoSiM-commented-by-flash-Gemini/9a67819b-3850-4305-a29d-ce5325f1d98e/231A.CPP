/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem from Codeforces, determining
 * how many problems a team of three friends will implement. A problem is
 * implemented if at least two of the three friends are sure about the solution.
 *
 * Algorithm: Iterative problem evaluation.
 * 1. Read the total number of problems `n`.
 * 2. Initialize a counter `score` to 0 for implemented problems.
 * 3. For each problem:
 *    a. Read three integers (0 or 1) representing each friend's confidence.
 *    b. Sum these three integers.
 *    c. If the sum is 2 or more, increment `score`.
 * 4. Print the final `score`.
 * Time Complexity: O(N) where N is the number of problems, as each problem is processed in constant time.
 * Space Complexity: O(1) for storing variables.
 */

// problem link - https://codeforces.com/problemset/problem/231/A
// sumanth2303
// Team - 231A
// ACCEPTED
// One day three best friends Petya, Vasya and Tonya decided to form a team and
// take part in programming contest..
#include <bits/stdc++.h>
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
  // Functional Utility: 'petya', 'vasya', 'tonya' store confidence for a problem; 'n' is problem count; 'score' tracks implemented problems.
  int petya, vasya, tonya, n, score = 0;
  // Functional Utility: Reads the total number of problems.
  cin >> n;
  // Block Logic: Iterates through each problem to determine if it will be implemented.
  // Invariant: `n` decrements, representing remaining problems to process. `score` accumulates implemented problems.
  while (n--) {
    // Functional Utility: Reads confidence levels for the three friends for the current problem.
    cin >> petya >> vasya >> tonya;
    // Block Logic: Checks if at least two friends are confident about the solution.
    // Pre-condition: `petya`, `vasya`, `tonya` are 0 or 1.
    if (petya + vasya + tonya >= 2) {
      score++; // Functional Utility: Increments the count of implemented problems.
    }
  }
  // Functional Utility: Prints the final total count of implemented problems.
  cout << score << endl;
  return 0;
}
