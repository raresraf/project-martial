/**
 * @file 282A.CPP
 * @brief Interprets a simplified programming language with increment/decrement operations
 *        on a single variable 'x', supporting both prefix and postfix notations.
 * Algorithm: Processes 'n' commands sequentially. Each command is a 3-character string.
 *            The code uses direct string character access and comparison to determine
 *            if the operation is an increment ("++X", "X++") or a decrement ("--X", "X--").
 * Time Complexity: O(N) where N is the number of operations, as each command is processed
 *                  with constant time string character access and comparison.
 * Space Complexity: O(1) for storing variables and the current command string.
 */
#include <iostream>
#include <string> // Required for std::string

using namespace std;

/**
 * @brief Main function to simulate a basic programming language's arithmetic operations.
 * Functional Utility: Reads a series of commands and updates a single integer variable 'x'
 *                     based on whether the command is an increment or decrement operation.
 */
int main() {
  int n; // Stores the total number of operations to be performed.
  cin >> n; // Reads the number of operations from standard input.
  
  int x = 0; // The core variable, initialized to 0, on which operations are performed.
  string s;  // String to store each command read from input.
  
  // Block Logic: Loops 'n' times, processing one command in each iteration.
  // Invariant: 'x' is updated according to the current command string 's'.
  while (n--) {
    cin >> s; // Reads the current 3-character command string.
    
    // Block Logic: Checks for increment operations ("++X" or "X++").
    // Functional Utility: If an increment pattern is matched, 'x' is increased by 1.
    if (s[0] == '+' && s[1] == '+' && s[2] == 'X') {
      x += 1;
    } else if (s[0] == 'X' && s[1] == '+' && s[2] == '+') {
      x += 1;
    } 
    // Block Logic: Checks for decrement operations ("X--" or "--X").
    // Functional Utility: If a decrement pattern is matched, 'x' is decreased by 1.
    else if (s[0] == 'X' && s[1] == '-' && s[2] == '-') {
      x -= 1;
    } else if (s[0] == '-' && s[1] == '-' && s[2] == 'X') {
      x -= 1;
    }
  }
  
  // Functional Utility: Prints the final accumulated value of 'x' to standard output.
  cout << x;
  return 0; // Indicates successful program termination.
}