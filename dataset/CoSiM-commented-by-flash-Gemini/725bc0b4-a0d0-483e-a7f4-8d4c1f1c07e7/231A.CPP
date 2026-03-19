/**
 * @file 231A.CPP
 * @brief Solution to Codeforces Problem 231A: "Team".
 * Determines the number of problems a team can solve. A problem is solvable if
 * at least two out of three friends are sure about the solution.
 * Algorithm: For each problem, iterate through the three friends' opinions. Count
 * how many friends are sure (opinion is 1). If this count is 2 or more, increment
 * the total number of solvable problems.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream.
using namespace std;

/**
 * @brief Main function to solve the "Team" problem.
 * Reads the number of problems, then for each problem, reads the certainty of three friends,
 * and calculates the total number of problems that can be solved.
 */
int main() {
  // Declare variables:
  // 'soal': total number of problems to evaluate.
  // 'yakin': count of friends who are sure about the current problem's solution (initialized to 0 per problem).
  // 'solusi': total count of problems that can be solved (initialized to 0).
  int soal, yakin = 0, solusi = 0;
  cin >> soal; // Reads the total number of problems.

  /**
   * Block Logic: Outer loop iterates through each problem.
   * Pre-condition: 'soal' specifies the number of problems. 'solusi' is 0.
   * Invariant: After each outer iteration, one problem has been fully evaluated, and 'solusi'
   *            accurately reflects the number of solvable problems found so far.
   */
  for (int i = 0; i < soal; i++) {
    int input; // Temporary variable to store the certainty of each friend.
    yakin = 0; // Reset 'yakin' counter for the new problem.

    /**
     * Block Logic: Inner loop iterates through the three friends for the current problem.
     * Pre-condition: 'input' is 0 or 1.
     * Invariant: After each inner iteration, 'yakin' counts how many friends processed so far are sure.
     */
    for (int j = 0; j < 3; j++) { // Loop for 3 friends.
      cin >> input; // Reads the certainty (0 or 1) of the current friend.
      if (input == 1) {
        yakin++; // If the friend is sure, increment 'yakin'.
      }
    }
    
    /**
     * Block Logic: Checks if at least two friends are sure about the solution.
     * If 'yakin' is 2 or more, the problem can be solved.
     */
    if (yakin >= 2) {
      solusi++; // Increment the total count of solvable problems.
    }
    // 'yakin' is reset to 0 at the start of the next outer loop iteration.
  }
  cout << solusi; // Prints the final count of problems that can be solved.

  return 0;
}
