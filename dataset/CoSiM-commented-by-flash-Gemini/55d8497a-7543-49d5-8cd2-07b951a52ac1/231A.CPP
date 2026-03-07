/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem that counts the number of "sure" problems.
 * This program reads 'n' problems, and for each problem, it reads the opinions of three friends.
 * A problem is considered "sure" if at least two friends are sure about the solution.
 * The program then outputs the total count of "sure" problems.
 * Algorithm: Iterative problem evaluation.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as only a few variables are used to store counts and temporary inputs.
 */
#include <iostream>

using std::cin;
using std::cout;

int main(void) {
  // Declares an integer 'n' to store the total number of problems.
  int n;
  // Declares an array 'sures' to temporarily store the 'sureness' of three friends for a problem.
  int sures[] = {0, 0, 0};

  // Reads the total number of problems from standard input.
  cin >> n;
  
  // Initializes 'count' to track the current problem being processed, acting as an iterator.
  int count = 0;
  // Initializes 'solutionCount' to accumulate the total number of problems where at least two friends are sure.
  int solutionCount = 0;

  /**
   * Block Logic: Iterates through each problem to evaluate if it's a "sure" solution.
   * Invariant: At the start of each iteration, 'count' represents the number of problems processed so far,
   * and 'solutionCount' holds the number of "sure" problems found among those processed.
   */
  while (count++ < n) {
    // Reads the sureness opinions of three friends for the current problem.
    cin >> sures[0] >> sures[1] >> sures[2];
    // Initializes 'sureCount' to tally the number of friends sure about the current problem's solution.
    int sureCount = 0;
    
    /**
     * Block Logic: Evaluates the opinions of the three friends for the current problem.
     * Invariant: At the start of each iteration, 'i' points to the current friend's opinion being checked.
     * Post-condition: 'sureCount' contains the total number of friends sure about the current problem.
     */
    for (int i = 0; i < 3; i++) {
      // Increments 'sureCount' if the current friend is sure (indicated by a non-zero value).
      if (sures[i]) {
        sureCount++;
      }
    }
    
    // Checks if at least two friends are sure about the current problem's solution.
    if (sureCount >= 2) {
      // If two or more friends are sure, increments the total count of "sure" problems.
      solutionCount++;
    }
  }

  // Prints the final total count of "sure" problems to standard output.
  cout << solutionCount;

  return 0;
}