/**
 * @file 282A.CPP
 * @brief Implements a simple counter based on increment/decrement statements.
 *
 * This program reads an integer 'n' followed by 'n' statements. Each statement
 * is expected to contain either "++" (increment operation) or "--" (decrement operation).
 * The program initializes a counter 'x' to 0 and updates it based on these operations.
 *
 * Algorithm: Iterative processing of 'n' input statements. String substring
 * comparison is used to identify increment/decrement operations.
 * Time Complexity: O(N * L), where N is the number of statements and L is the
 * average length of a statement (due to substring operations). Given L is constant (3),
 * the complexity is effectively O(N).
 * Space Complexity: O(L) for storing statement strings, effectively O(1).
 */
#include <iostream>
#include <string>

// Functional Utility: Using declarations to simplify access to standard library components.
using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  /**
   * @brief Main function: Processes 'n' increment/decrement statements and outputs the final counter value.
   * @return 0 if the program executes successfully.
   */
  int n, x = 0; // 'n': number of statements, 'x': the counter initialized to 0.
  cin >> n;

  string statement, lowSubstr, highSubstr;
  /**
   * @brief Loop through 'n' statements to update the counter 'x'.
   * @pre 'n' is a positive integer.
   * @invariant At the beginning of each iteration, 'x' holds the accumulated value
   *            from previously processed statements.
   */
  for (int i = 0; i != n; ++i) { // Loop 'n' times, once for each statement.
    cin >> statement; // Read the current statement.
    // Functional Utility: Extract substrings to check for "++" or "--".
    // This approach handles cases like "++X" or "X++", or "--X" or "X--".
    lowSubstr = statement.substr(0, 2);  // Check first two characters.
    highSubstr = statement.substr(1, 3); // Check middle two characters for "X++" or "X--".

    /**
     * @brief Conditional block to update the counter 'x' based on the statement.
     * @pre 'statement' has been read and its relevant substrings extracted.
     */
    if (lowSubstr == "++" || highSubstr == "++") {
      x++; // Increment 'x' if "++" is found.
    } else if (lowSubstr == "--" || highSubstr == "--") {
      x--; // Decrement 'x' if "--" is found.
    } else {
      // Invariant: If neither "++" nor "--" is found (unexpected input), continue to the next iteration.
      continue;
    }
  }
  cout << x; // Output the final value of the counter 'x'.
  return 0;
}
