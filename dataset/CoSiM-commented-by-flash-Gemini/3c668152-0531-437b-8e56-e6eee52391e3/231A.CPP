/**
 * @file 231A.CPP
 * @brief This program solves a problem where it processes 't' sets of three binary values (0 or 1).
 * For each set, it determines if at least two of the three values are equal to 1.
 * If this condition is met, a counter is incremented. Finally, the total count is printed.
 * This is a straightforward problem testing conditional logic and input processing, common in competitive programming.
 * Algorithm: Iterative processing with simple conditional summation.
 * Time Complexity: O(T), where T is the number of sets, as it iterates through each set once.
 * Space Complexity: O(1) for storing a few variables.
 */
#include <iostream> // Includes the standard input/output stream library.
using namespace std;
int main() {
  int t; // Declares 't' for the number of sets (or test cases).
  cin >> t; // Reads the number of sets 't'.
  int count = 0; // Initializes 'count' to store the total number of sets satisfying the condition.
  /**
   * Block Logic: Loop iterates 't' times, once for each set of three binary values.
   */
  for (int i = 0; i < t; i++) {
    int a, b, c; // Declares three integer variables to hold the binary values for the current set.
    cin >> a >> b >> c; // Reads the three binary values.
    /**
     * Conditional Logic: Checks if the sum of the three binary values is greater than or equal to 2.
     * This elegantly determines if at least two of the three values are 1.
     */
    if (a + b + c >= 2)
      count++; // Increments the counter if the condition is satisfied.
  }
  cout << count << endl; // Prints the final count, followed by a newline.
}
