/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem: "Team".
 * This program reads an integer 't' representing the number of problems. For each problem,
 * it reads three integers representing the confidence of three friends. If at least
 * two friends are confident (sum of confidence >= 2), a counter 'num' is incremented.
 * Finally, it prints the total number of problems solved.
 * Algorithm: Iterative conditional counting based on a sum threshold.
 * Time Complexity: O(T) where T is the number of problems.
 * Space Complexity: O(1)
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function where the program execution begins.
 * Handles input for the number of problems and friend confidences,
 * then calculates and displays the total problems solved.
 */
int main() {
  int t; // Number of problems to consider.
  int array[3]; // Array to store confidence levels of three friends (0 or 1).
  int num = 0; // Counter for problems solved.
  // Reads the total number of problems 't'.
  cin >> t;
  /**
   * Block Logic: Iterates through each problem to determine if it's solved.
   * Invariant: 't' decreases with each iteration, controlling the loop execution.
   * Pre-condition: 't' is a non-negative integer.
   */
  for (; t > 0; t--) {
    // Reads the confidence levels of the three friends for the current problem.
    cin >> array[0] >> array[1] >> array[2];
    /**
     * Block Logic: Checks if at least two friends are confident.
     * Pre-condition: 'array' contains confidence levels (0 or 1).
     * Invariant: 'num' counts problems where confidence sum is 2 or more.
     */
    if (array[0] + array[1] + array[2] >= 2) {
      // Increments the counter if the sum of confidence is 2 or more.
      num++;
    }
  }
  // Prints the total number of problems solved.
  cout << num << endl;
  return 0;
}
