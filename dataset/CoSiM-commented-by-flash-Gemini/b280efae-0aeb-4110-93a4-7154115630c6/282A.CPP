/**
 * @file 282A.cpp
 * @brief Solves the "Bit++" problem from Codeforces (Problem 282A).
 *        This program simulates a sequence of operations on a variable 'x',
 *        where each operation can either increment or decrement 'x'.
 *        The operations are given as strings, and the program determines
 *        the final value of 'x'.
 * Algorithm: Counter Simulation / String Parsing
 * Time Complexity: O(N * L), where N is the number of operations and L is the
 *                  maximum length of an operation string (due to `string::npos`).
 *                  If L is considered constant (e.g., "X++" or "--X"), then O(N).
 * Space Complexity: O(L) for storing the operation string. If L is constant, then O(1).
 */

// 282A.cpp
// A. Bit++
// Author: Ikki
// May 31, 2015

#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for using std::string.
using namespace std;

/**
 * @brief Main function to process a series of operations and calculate
 *        the final value of a variable 'x'.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Declare 'n' for the number of operations and 'x' for the variable to be manipulated, initialized to 0.
  int n, x = 0;
  // Read the number of operations from standard input.
  cin >> n;
  
  /**
   * Block Logic: Processes each of the 'n' operations.
   * Invariant: 'x' reflects the result of all operations processed so far.
   */
  for (int i = 0; i < n; i++) {
    string str; // Declare a string to store the current operation.
    // Read the operation string from standard input.
    cin >> str;
    
    // Block Logic: Determines whether to increment or decrement 'x' based on the operation string.
    // Pre-condition: 'str' contains an operation string like "X++", "++X", "X--", or "--X".
    // Post-condition: 'x' is either incremented or decremented.
    // Check if the string contains a '+' character.
    // string::npos is returned by find() if the character is not found.
    if (str.find('+') != string::npos)
      x++; // If '+' is found, increment 'x'.
    else
      x--; // Otherwise (if '-' is implied), decrement 'x'.
  }
  
  // Print the final value of 'x' followed by a newline.
  cout << x << endl;
  return 0; // Indicate successful program termination.
}