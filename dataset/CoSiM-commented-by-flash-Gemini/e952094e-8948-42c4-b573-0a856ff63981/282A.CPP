/**
 * @file 282A.CPP
 * @brief Implements a simple programming language interpreter that processes
 *        increment and decrement operations on a single variable 'x'.
 * Algorithm: Processes 'n' commands sequentially, updating the value of 'x'
 *            based on string comparisons for increment ("X++", "++X") or
 *            decrement ("X--", "--X") operations.
 * Time Complexity: O(N * L), where N is the number of commands and L is the
 *                  average length of a command string, due to string comparisons.
 * Space Complexity: O(L) for storing the command string.
 */
#include <iostream>
#include <string> // Required for string manipulation

using namespace std;

/**
 * @brief Main function to process a series of increment/decrement commands.
 * Functional Utility: Reads a number 'n' indicating the count of operations,
 *                     then processes 'n' command strings, updating an integer 'x'.
 */
int main() {
  int n; // Variable to store the number of operations to perform.
  int x = 0; // The integer variable that will be incremented or decremented. Initialized to 0.
  
  // Block Logic: Reads the total number of operations from standard input.
  cin >> n;

  // Block Logic: Iterates 'n' times, processing one command in each iteration.
  // Pre-condition: 'n' is a non-negative integer.
  // Invariant: 'x' is updated according to the command in each iteration.
  for (int i = 0; i < n; i++) {
    string command;
    // Reads the current command string (e.g., "X++", "--X").
    cin >> command;
    
    // Block Logic: Checks if the command is an increment operation.
    // Functional Utility: If the command matches "X++" or "++X", the value of 'x' is increased by 1.
    if (command.compare("X++") == 0 || command.compare("++X") == 0) {
      x = x + 1; // Increment x.
    }
    
    // Block Logic: Checks if the command is a decrement operation.
    // Functional Utility: If the command matches "X--" or "--X", the value of 'x' is decreased by 1.
    if (command.compare("X--") == 0 || command.compare("--X") == 0) {
      x = x - 1; // Decrement x.
    }
  }
  
  // Functional Utility: Prints the final value of 'x' followed by a newline.
  cout << x << "\n";
  return 0; // Indicates successful program termination.
}