/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem, which involves incrementing and decrementing a variable.
 * @details This program simulates a series of operations on a variable, initialized to zero.
 * The operations are given as strings, such as "++X" or "X--", and the program
 * calculates the final value of the variable after all operations are performed.
 *
 * @section algorithm Algorithm
 * 1. Read the total number of operations, `a`.
 * 2. Initialize a variable, `n`, to 0. This will be the variable that is modified.
 * 3. Loop `a` times to process each operation.
 *    a. In each iteration, read the operation string `c`.
 *    b. Check if the string `c` corresponds to an increment operation ("++X" or "X++").
 *       If it is, increment `n`.
 *    c. If it's not an increment operation, it must be a decrement operation, so decrement `n`.
 * 4. After the loop completes, print the final value of `n`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(a), where `a` is the number of operations. The program iterates through each operation once.
 * - Space Complexity: O(1) for the variable storage. The space for the string `c` is reused in each iteration.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to execute the sequence of operations.
 * @return 0 on successful execution.
 */
int main() {
  int a;      // Variable to store the number of operations.
  int n = 0;  // The variable to be incremented or decremented, initialized to 0.
  string c;   // String to store the operation.

  // Read the total number of operations.
  cin >> a;

  // Loop through all the operations.
  for (int i = 0; i < a; i++) {
    cin >> c;

    // Check if the operation is an increment.
    if (c == "++X" || c == "X++") {
      n++;
    } else { // Otherwise, it's a decrement.
      n--;
    }
  }

  // Print the final value of the variable.
  cout << n;
  
  return 0;
}