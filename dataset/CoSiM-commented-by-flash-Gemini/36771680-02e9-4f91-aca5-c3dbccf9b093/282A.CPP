/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem: "Bit++".
 *
 * This problem involves processing a sequence of statements that modify a single
 * integer variable, 'x'. Each statement consists of an operation ("++X", "X++", "--X", "X--")
 * and the program determines the final value of 'x' after all operations.
 *
 * Algorithm: Iterative variable modification based on string operations.
 * Time Complexity: O(N * L), where N is the number of statements and L is the average length of a statement string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Variable Declarations:
  // ch: Stores the current operation statement as a string.
  // a: Represents the number of operations to be performed in a test case.
  // i: Loop counter for iterating through operations.
  // x: The integer variable whose value is modified by the operations.
  string ch;
  int a, i, x;

  // Block Logic: Main loop for processing multiple test cases.
  // Invariant: Each iteration processes a complete set of operations for a single test case.
  while (cin >> a) { // Precondition: Reads the number of operations for the current test case.
    x = 0;           // Functional Utility: Initializes 'x' to 0 for each new test case.

    // Block Logic: Iterates through each operation for the current test case.
    // Invariant: For each iteration, `i` represents the current operation count (1-indexed).
    for (i = 1; i <= a; i++) {
      // Input: Reads the operation statement.
      cin >> ch;
      // Block Logic: Checks if the operation is an increment.
      // Precondition: `ch` contains a valid operation string.
      if (ch == "++X" || ch == "X++") {
        x++; // Inline: Increments 'x' by 1.
      } else if (ch == "--X" || ch == "X--") { // Block Logic: Checks if the operation is a decrement.
        x--; // Inline: Decrements 'x' by 1.
      }
    }
    // Output: Prints the final value of 'x' for the current test case.
    cout << x << endl;
  }
  return 0; // Functional Utility: Indicates successful program execution.
}
