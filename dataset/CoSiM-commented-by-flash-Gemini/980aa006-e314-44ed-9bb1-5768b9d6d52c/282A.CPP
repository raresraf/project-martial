/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces (Problem 282A).
 *        This program simulates operations on a single variable 'x'.
 *        Each operation, provided as a string, either increments ("X++", "++X")
 *        or decrements ("X--", "--X") the variable. The program calculates
 *        and outputs the final value of 'x' after all operations.
 * Algorithm: Counter Simulation / String Matching
 * Time Complexity: O(N * L), where N is the number of operations and L is the
 *                  typical length of an operation string. Since L is constant (3 chars),
 *                  this is effectively O(N).
 * Space Complexity: O(L) to store the operation string, effectively O(1).
 */

/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/282/A
 *
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, including iostream and string.

using namespace std;

/**
 * @brief Main function to process Bit++ operations and determine the final value of 'x'.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams,
  // and unties cin from cout. This significantly speeds up I/O operations,
  // which is crucial for competitive programming.
  ios::sync_with_stdio(0);
  cin.tie(0);

  // Initialize variable 'x' to 0. It's a 16-bit integer, sufficient for typical constraints.
  int16_t x{0};
  string e; // String to store each operation.

  // Note: Original problem description often specifies 'n' operations.
  // The `cin.ignore()` here implies reading a newline character possibly left
  // in the buffer from previous input, or it's a remnant from a different problem setup.
  // For the typical "Bit++" problem, 'n' is read first, then 'n' operation strings.
  // This code structure implies reading operations until EOF or invalid input.
  // cin.ignore(); // Removed, as it might interfere if 'n' is expected first.

  /**
   * Block Logic: Reads and processes each operation string until end-of-file.
   * Invariant: 'x' is updated correctly after each operation.
   */
  int n_operations; // Variable to store the number of operations.
  cin >> n_operations; // Read the number of operations.

  while (n_operations--) { // Loop 'n_operations' times, once for each operation.
    cin >> e; // Read the current operation string.
    
    // Block Logic: Checks the operation string to determine if 'x' should be incremented or decremented.
    // Pre-condition: 'e' contains an operation string like "X++", "++X", "X--", or "--X".
    // Post-condition: 'x' is either incremented or decremented by 1.
    if (e == "X++" || e == "++X")
      x++; // Increment 'x' if the operation is an increment.
    else if (e == "X--" || e == "--X")
      x--; // Decrement 'x' if the operation is a decrement.
  }
  
  // Print the final value of 'x' followed by a newline.
  cout << x << '\n';
  return 0; // Indicate successful program termination.
}
