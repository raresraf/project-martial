/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on member confidence.
 * @details The problem is from Codeforces: http://codeforces.com/problemset/problem/231/A
 *
 * A team of three members is given a number of problems. For each problem, each
 * member indicates if they are sure about the solution (1 for sure, 0 for not sure).
 * The team will only solve a problem if at least two members are sure. This program
 * calculates the total number of problems the team will solve.
 *
 * @algorithm Iterative Counting
 * @complexity
 * Time: O(N), where N is the number of problems, as we process each problem once.
 * Space: O(1), as we only use a few variables to store the counts.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to read problem data and calculate the result.
 */
int main() {
  // testCases: the total number of problems to evaluate.
  // Petya, Vasya, Tonya: variables to store the confidence of each team member (1 or 0).
  // TeamScore: a counter for the number of problems the team will solve.
  int testCases, Petya, Vasya, Tonya, TeamScore = 0;
  cin >> testCases;

  // Pre-condition: TeamScore is 0.
  // Block Logic: Loop through each problem.
  while (testCases--) {
    // Read the confidence levels for the current problem.
    cin >> Petya >> Vasya >> Tonya;

    // Invariant: The sum represents the number of members confident about the solution.
    // Block Logic: If two or more members are confident, the team solves the problem.
    if (Petya + Vasya + Tonya >= 2) {
      TeamScore += 1;
    }
  }

  // Output the total number of problems that will be solved.
  cout << TeamScore << endl;

  return 0;
}
