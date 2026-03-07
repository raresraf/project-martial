/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem where the goal is to count
 * how many problems can be implemented given three participants' opinions.
 * A problem is considered implementable if at least two out of the three participants
 * are "sure" (represented by a non-zero value).
 * Algorithm: Iterative conditional checking and counting.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), for storing a few integer variables.
 */
#include <iostream>
#include <regex> // Included but not used, typical in competitive programming for convenience.
#include <string> // Included but not used, typical in competitive programming for convenience.

/**
 * @brief Determines if a problem can be implemented based on three opinions.
 *
 * This function takes three integer inputs (representing opinions) and returns
 * true if at least two of them are non-zero (indicating "sure" votes).
 *
 * @param a Opinion of the first participant (non-zero if sure).
 * @param b Opinion of the second participant (non-zero if sure).
 * @param c Opinion of the third participant (non-zero if sure).
 * @return bool: True if at least two participants are sure, false otherwise.
 */
bool implement(int a, int b, int c) {
  // Checks all combinations where at least two participants are sure.
  if (a && b || a && c || b && c) {
    // If the condition is met, the problem is implementable.
    return true;
  } else {
    // Otherwise, the problem is not implementable based on these opinions.
    return false;
  }
}

int main() {
  // Declares 'n' to store the total number of problems.
  int n;
  // Declares 'a', 'b', 'c' to store the opinions for each problem.
  int a, b, c;

  // Initializes 'implementCounter' to count problems that can be implemented.
  int implementCounter = 0;

  // Reads the total number of problems 'n' from standard input.
  std::cin >> n;

  /**
   * Block Logic: Iterates 'n' times, once for each problem.
   * Invariant: At the start of each iteration, 'i' represents the current problem index.
   * Post-condition: After the loop, 'implementCounter' holds the total count of implementable problems.
   */
  for (int i = 0; i < n; i++) {
    // Reads the three opinions for the current problem.
    std::cin >> a >> b >> c;

    // Calls the 'implement' function to check if the current problem is implementable.
    if (implement(a, b, c)) {
      // If true, increments the counter for implementable problems.
      implementCounter++;
    }
  }

  // Prints the final count of implementable problems to standard output.
  std::cout << implementCounter << std::endl;

  return 0;
}
