/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will implement based on judge agreement.
 *        A problem is implemented if at least two out of three judges agree.
 *
 * Algorithm: Iterative summation of judge agreements and conditional counting.
 * Time Complexity: O(N) where N is the number of problems, due to a single linear pass.
 * Space Complexity: O(N) for storing the sum of agreements for each problem.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

// Array to store the sum of agreements for each problem. Max 1111 problems.
int s[1111];

/**
 * @brief Main function to calculate the number of problems to be implemented.
 * @pre Input consists of an integer 'n' (total problems), followed by 'n' lines,
 *      each containing three binary integers (a, b, c) representing judge agreements.
 * @post Prints the total count of problems that will be implemented.
 */
int main() {
  int n, a, b, c, sum = 0; // 'n': total problems, 'a, b, c': judge agreements, 'sum': count of implemented problems.

  // Functional Utility: Reads the total number of problems.
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to read judge agreements, sum them, and check for implementation.
   * Invariant: In each iteration, 's[i]' stores the sum of agreements for the current problem, and 'sum'
   *            is incremented if 's[i]' indicates implementation (s[i] > 1).
   */
  for (int i = 0; i < n; ++i) {
    // Functional Utility: Reads the three judge agreements for the current problem.
    cin >> a >> b >> c;
    s[i] = a + b + c; // Calculate the sum of agreements for the current problem.

    /**
     * Block Logic: Checks if the current problem will be implemented.
     * Pre-condition: 's[i]' holds the sum of agreements for the current problem.
     * Invariant: If 's[i]' is greater than 1 (meaning at least two judges agreed), 'sum' is incremented.
     */
    if (s[i] > 1)
      ++sum; // Increment the count of implemented problems.
  }

  // Functional Utility: Prints the final count of problems to be implemented.
  cout << sum;
  // Indicates successful program execution.
  return 0;
}
