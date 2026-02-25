/**
 * @file 231A.CPP
 * @brief This program solves a problem where it processes 'n' sets of three binary values (0 or 1).
 * For each set, it determines if at least two of the three values are equal to 1.
 * If this condition is met, a counter is incremented. Finally, the total count is printed.
 * This is a straightforward problem testing conditional logic and input processing, common in competitive programming.
 * Algorithm: Iterative processing with simple conditional summation.
 * Time Complexity: O(N), where N is the number of sets, as it iterates through each set once.
 * Space Complexity: O(1) for storing a few variables and a small array.
 */
#include <iostream> // Includes the standard input/output stream library.
using namespace std;
int main() {
  int n, count1 = 0, a[3]; // Declares 'n' for the number of sets, 'count1' to store the total count, and 'a' as an array to hold the three binary values.

  cin >> n; // Reads the number of sets 'n'.
  /**
   * Block Logic: Outer loop iterates 'n' times, once for each set of binary values.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Inner loop reads three binary values into the array 'a' for the current set.
     */
    for (int j = 0; j < 3; j++) {
      cin >> a[j];
    }
    /**
     * Conditional Logic: Checks if at least two of the three binary values in array 'a' are 1.
     * The condition `(a[0] == 1 && a[1] == 1) || (a[1] == 1 && a[2] == 1) || (a[0] == 1 && a[1] == 1 && a[2] == 1) || (a[0] == 1 && a[2] == 1)`
     * can be simplified to `(a[0] + a[1] + a[2]) >= 2`.
     * If true, it means the condition (at least two 1s) is met, and the counter is incremented.
     */
    if ((a[0] == 1 && a[1] == 1) || (a[1] == 1 && a[2] == 1) ||
        (a[0] == 1 && a[1] == 1 && a[2] == 1) || (a[0] == 1 && a[2] == 1)) {
      count1++; // Increments the counter if the condition is satisfied.
    }
  }
  cout << endl << count1 << endl; // Prints a newline, then the final count, followed by another newline.
}
