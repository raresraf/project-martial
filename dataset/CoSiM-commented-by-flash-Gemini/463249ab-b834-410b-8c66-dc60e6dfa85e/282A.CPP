/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem (Codeforces 282A).
 *
 * This program simulates a simple programming language with a single integer
 * variable 'x' and two types of operations: increment ('X++' or '++X') and
 * decrement ('X--' or '--X'). It reads a sequence of 'n' operations and
 * calculates the final value of 'x' starting from 0.
 *
 * Algorithm: Direct simulation. Each operation string is read and the variable 'x'
 * is incremented or decremented accordingly.
 * Time Complexity: O(N * L) where N is the number of operations and L is the maximum
 * length of an operation string (constant, e.g., 3). This simplifies to O(N).
 * Space Complexity: O(L) to store the operation string for each read (constant space).
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to simulate Bit++ operations and calculate the final value of 'x'.
 *
 * Reads the number of operations 'n' and then processes 'n' operation strings.
 * It initializes 'x' to 0 and updates it based on increment or decrement operations.
 * Finally, it prints the resultant value of 'x'.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare a string 's' to temporarily store each operation.
  string s;
  // Declare an integer 'n' for the number of operations and 'x' for the variable to be modified, initialized to 0.
  int n, x = 0;
  // Read the number of operations 'n' from standard input.
  cin >> n;
  /**
   * @brief Block Logic: Processes 'n' operations.
   *
   * Iterates 'n' times, reading each operation string and updating the variable 'x' accordingly.
   * Pre-condition: 'n' specifies the total number of operations.
   * Invariant: 'x' is updated based on whether the operation is an increment or decrement.
   */
  for (int i = 0; i < n; i++) {
    // Read the operation string for the current iteration.
    cin >> s;
    // Pre-condition: 's' contains a valid operation string ("X++", "++X", "X--", or "--X").
    // Block Logic: Checks if the operation is an increment.
    if (s == "X++" || s == "++X") {
      x++; // Increment 'x'.
    }
    // Block Logic: Checks if the operation is a decrement.
    else if (s == "X--" || s == "--X") {
      x--; // Decrement 'x'.
    }
  }
  // Print the final value of 'x' followed by a newline.
  cout << x << endl;
  return 0;
}