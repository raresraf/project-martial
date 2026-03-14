/**
 * @64533096-8d98-4208-9b17-c277f6b29500/231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A). It reads 'n' problems,
 * and for each problem, three binary inputs (0 or 1) representing whether each of three
 * friends is sure about the solution. A problem is counted if at least two friends are
 * sure about its solution.
 *
 * Algorithm: Iterative processing of problems, checking the sum of binary inputs for a majority.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input (vector 'v' is declared but unused).
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <vector>   // For using std::vector (declared but unused).
using namespace std; // Uses the standard namespace.

int n, e; // @brief 'n' for the total number of problems, 'e' for individual friend's certainty.
int solved; // @brief Counter for friends who are sure about a problem's solution.
int result = 0; // @brief Accumulator for the total number of solvable problems.
vector<int> v; // @brief Declared but unused vector.

/**
 * @brief Main function to process problem certainty and count solvable problems.
 * Reads 'n' problems, three certainty values for each, and increments a counter
 * if at least two friends are sure.
 * @return 0 on successful execution.
 */
int main() {
  cin >> n; // Reads the number of problems from standard input.
  
  /**
   * Block Logic: Iterates through each of the 'n' problems.
   * Invariant: Each problem is evaluated to determine if it can be solved.
   */
  for (int i = 0; i < n; i++) {
    solved = 0; // Resets the 'solved' counter for each new problem.
    
    /**
     * Block Logic: Reads the certainty for three friends for the current problem.
     * Invariant: `solved` counts how many friends are sure about the current problem.
     */
    for (int i = 0; i < 3; i++) { // Loops three times, once for each friend.
      cin >> e; // Reads friend's certainty (0 or 1).
      if (e == 1) { // Inline: If the friend is sure (e=1).
        solved++; // Increments the count of sure friends.
      }
    }
    
    /**
     * Block Logic: Checks if the current problem is solvable (at least two friends are sure).
     * Pre-condition: `solved` contains the count of sure friends for the current problem.
     * Invariant: `result` is incremented if the current problem meets the solvability criteria.
     */
    if (solved >= 2) { // If two or more friends are sure.
      result++; // Increments the total count of solvable problems.
    }
  }
  cout << result; // Prints the final count of solvable problems.
  return 0; // Indicates successful program termination.
}
