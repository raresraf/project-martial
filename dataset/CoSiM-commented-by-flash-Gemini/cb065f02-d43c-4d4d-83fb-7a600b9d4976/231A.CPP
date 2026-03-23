/**
 * @file 231A.CPP
 * @brief This competitive programming solution addresses a problem where three friends
 *        decide to implement a problem if at least two of them are sure about the solution.
 *        Given a list of problems and each friend's confidence (represented as a boolean),
 *        the task is to count how many problems they will implement.
 * Algorithm: Iterative Conditional Check.
 *            1. Read the number of problems `n`.
 *            2. For each problem, read three boolean values representing each friend's confidence.
 *            3. Check specific combinations of two friends being sure. If any such pair is found,
 *               it implies at least two friends are sure, and a counter is incremented.
 *            4. Finally, print the total count of implemented problems.
 *            Note: A more concise and robust way to check "at least two" for three boolean values
 *            would typically involve summing the boolean values (treating true as 1, false as 0)
 *            and checking if the sum is >= 2. This implementation uses explicit `if-else if` for pairs.
 * Time Complexity: O(N), where N is the number of problems. Each problem is processed
 *                  with a constant number of operations (reading three booleans and a few comparisons).
 * Space Complexity: O(1), as only a few fixed-size integer and boolean variables are used.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std;

/**
 * @brief Main function of the program. Reads problem data, counts implementable problems, and prints the result.
 * Functional Utility: Serves as the entry point for execution, solving the problem of counting
 *                     problems with sufficient team confidence.
 */
int main() {
  int n = 0; // Declares an integer to store the number of problems.
  int x = 0; // Counter for problems that will be implemented.
  bool arr[3]; // Boolean array to store the confidence levels of the three friends for the current problem.

  cin >> n; // Reads the number of problems.

  // Block Logic: Iterates through each problem to determine if it will be implemented.
  // Invariant: `x` accurately reflects the number of implementable problems processed so far.
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the three boolean confidence values for the current problem.
    // Invariant: `arr` holds the confidence levels for friends 0, 1, and 2.
    for (int a = 0; a < 3; a++) {
      cin >> arr[a]; // Reads boolean (0 or 1, automatically converted to false/true) for each friend.
    }
    // Block Logic: Checks if at least two friends are sure about the solution.
    // This chain explicitly checks pairs. If multiple pairs are true (e.g., all three are true),
    // only the first true condition will increment `x`.
    if (arr[0] && arr[2] == true) { // Checks if friend 0 AND friend 2 are sure.
      x++;                          // If both are sure, increment the count.
    } else if (arr[1] && arr[0] == true) { // Checks if friend 1 AND friend 0 are sure.
      x++;                                 // If both are sure, increment the count.
    } else if (arr[1] && arr[2] == true) { // Checks if friend 1 AND friend 2 are sure.
      x++;                                 // If both are sure, increment the count.
    }
  }
  cout << x; // Prints the final total count of problems that will be implemented.
}
