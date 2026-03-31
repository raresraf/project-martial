/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 * This program counts the number of problems a team of three friends will implement.
 * A problem is implemented if at least two friends are sure about the solution.
 *
 * Algorithm: Iterative counting. For each problem, sum the confidence levels of the three friends
 * (where 1 means sure, 0 means not sure) and check if the sum is 2 or more.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <iostream> // Required for standard input/output streams (cin, cout)
using namespace std;

int main() {
  /**
   * @brief Main function to determine the number of problems a team will implement.
   * Reads the total number of problems, then for each problem, reads the confidence
   * of three friends, and increments a counter if at least two friends are confident.
   */
  int n; // @brief Total number of problems.
  cin >> n;
  
  int count = 0; // @brief Counter for problems the team will implement.
  
  // Block Logic: Iterate through each problem to assess team's ability to implement it.
  // Pre-condition: 'n' problems are expected to be processed.
  // Invariant: 'count' accurately reflects the number of implementable problems encountered so far.
  for (int i = 1; i <= n; i++) {
    int petya, vasya, tonya; // @brief Confidence levels of Petya, Vasya, and Tonya (1 for sure, 0 for not sure).
    // Inline: Read confidence levels for the current problem.
    cin >> petya >> vasya >> tonya;
    
    // Block Logic: Determine if at least two friends are sure about the solution.
    // If the sum of their confidence is 2 or more, the problem will be implemented.
    if (petya + vasya + tonya >= 2) {
      count++; // Inline: Increment the counter for implementable problems.
    }
  }
  // Block Logic: Output the total number of problems the team will implement.
  cout << count;
  return 0; // @brief Indicates successful program termination.
}