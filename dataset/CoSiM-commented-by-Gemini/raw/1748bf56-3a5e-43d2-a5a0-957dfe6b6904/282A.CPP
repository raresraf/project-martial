/**
 * @file 282A.CPP
 * @brief Solution for a problem involving simple increment/decrement operations on a variable.
 * @details This program reads a series of statements, each of which is either an increment
 * or a decrement operation on a variable, and calculates the final value of the variable.
 * The operations can be in prefix or postfix form (e.g., "++X" or "X++").
 *
 * @section algorithm Algorithm
 * 1. Initialize a variable `init` to 0. This variable represents the value being tracked.
 * 2. Read an integer `n`, which is the total number of operations to be performed.
 * 3. Loop `n` times to process each operation:
 *    a. Read the operation as a string (e.g., "++X", "X--").
 *    b. Use `strcmp` to compare the input string with the four possible operation strings:
 *       - If the string is "++X" or "X++", increment the `init` variable.
 *       - If the string is "--X" or "X--", decrement the `init` variable.
 * 4. After the loop finishes, print the final value of the `init` variable.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of operations. The program iterates through the `n` operations once.
 * - Space Complexity: O(1), as the space used is constant regardless of the input size (a few variables and a small character array).
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the program logic.
 * @return 0 on successful execution.
 */
int main() {
  // 'init' holds the value that will be modified by the operations.
  int init = 0, n;
  // 'str' will store each operation string, which is at most 3 characters long plus a null terminator.
  char str[4];
  
  // Read the number of operations.
  cin >> n;
  
  // Loop to process each of the 'n' operations.
  for (int i = 0; i < n; i++) {
    cin >> str;
    
    // Check for increment operations.
    if (!strcmp(str, "++X"))
      ++init;
    if (!strcmp(str, "X++"))
      init++;

    // Check for decrement operations.
    if (!strcmp(str, "--X"))
      --init;
    if (!strcmp(str, "X--"))
      init--;
  }
  
  // Print the final value of 'init'.
  cout << init << endl;
  return 0;
}
