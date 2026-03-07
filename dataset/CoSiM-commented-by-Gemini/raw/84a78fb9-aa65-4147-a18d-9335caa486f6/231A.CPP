/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from competitive programming contests.
 *
 * @details
 * The program determines the number of problems a three-person team will solve.
 * The team decides to solve a problem only if at least two members are confident
 * about the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, n.
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop n times, once for each problem.
 *    a. Read the confidence levels (0 or 1) of the three team members.
 *    b. Sum their confidence levels.
 *    c. If the sum is 2 or greater, it means at least two members are sure,
 *       so increment the counter.
 * 4. Print the final count of solvable problems.
 *
 * Time Complexity: O(N), where N is the number of problems, as the program iterates through each problem once.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n; // Read the total number of problems.
  int count = 0; // Counter for the number of problems the team will solve.
  
  /**
   * @brief Loop to iterate through each problem.
   * Pre-condition: 'n' holds the total number of problems.
   * Invariant: 'count' holds the number of problems solved among the first 'i-1' problems.
   */
  for (int i = 1; i <= n; i++) {
    int petya, vasya, tonya;
    // Read the confidence level of each of the three members for the current problem.
    cin >> petya >> vasya >> tonya;
    
    /**
     * @brief Check if the team will solve the problem.
     * The condition `petya + vasya + tonya >= 2` checks if at least two members
     * are sure about the solution (represented by a confidence level of 1).
     */
    if (petya + vasya + tonya >= 2) {
      count++;
    }
  }
  
  // Print the total number of problems the team will implement.
  cout << count;
  return 0;
}