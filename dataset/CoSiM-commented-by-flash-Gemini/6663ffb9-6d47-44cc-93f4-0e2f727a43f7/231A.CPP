/**
 * @6663ffb9-6d47-44cc-93f4-0e2f727a43f7/231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A).
 * It reads 'n' problems. For each problem, it takes three binary inputs (0 or 1)
 * representing whether each of three friends is sure about the solution.
 * A problem is considered solvable if at least two friends are sure about it.
 * The program counts and outputs the total number of solvable problems.
 *
 * Algorithm: Iterative processing of problems, checking the sum of binary inputs for a majority.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input.
 */
// By archit-dev, contest: Codeforces Round #143 (Div. 2), problem: (A) Team,
// Accepted, #, Copy
#include <bits/stdc++.h> // Includes many standard C++ library headers.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to process problem certainty and count solvable problems.
 * Reads 'n' problems, three certainty values for each, and increments a counter
 * if at least two friends are sure.
 * @return 0 on successful execution.
 */
int main() {

  int n; // Declares 'n', representing the total number of problems.
  int petya, vatsya, tonya; // Declares variables for each friend's certainty (0 or 1).
  int count = 0; // Accumulator for the total number of solvable problems.
  cin >> n; // Reads the number of problems from standard input.
  
  /**
   * Block Logic: Iterates through each of the 'n' problems.
   * For each problem, it reads the certainty of three friends and determines
   * if the problem is solvable (at least two friends are sure).
   * Invariant: `count` accurately reflects the number of solvable problems processed so far.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each problem.
    // Reads the certainty values for Petya, Vatsya, and Tonya for the current problem.
    cin >> petya >> vatsya >> tonya; 
    
    // Block Logic: Checks if the sum of certainties is 2 or more (meaning at least two friends are sure).
    if (petya + vatsya + tonya >= 2) { // Pre-condition: Sum of friends' certainties for the current problem.
      count++; // Increments the total count of solvable problems.
    } else {
      // Inline: No action needed if the problem is not solvable; loop continues to the next iteration.
      continue; 
    }
  }
  cout << count << "\n"; // Prints the final count of solvable problems, followed by a newline.
  return 0; // Indicates successful program termination.
}
