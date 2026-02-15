/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" competitive programming problem.
 * @details This program calculates the final value of a variable 'X' after a series
 * of increment ("++X", "X++") and decrement ("--X", "X--") operations.
 * The initial value of X is 0.
 *
 * @section algorithm Algorithm
 * The algorithm iterates through each operation and updates a counter:
 * 1. Read the total number of operations to be performed.
 * 2. Initialize a counter variable, `x_value`, to 0.
 * 3. Loop for the number of operations. In each iteration:
 *    a. Read the operation as a string.
 *    b. If the string matches an increment operation, increment `x_value`.
 *    c. Otherwise, it's a decrement operation, so decrement `x_value`.
 * 4. After the loop, print the final value of the counter.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of operations. String comparison takes constant time as the strings are short.
 * - Space Complexity: O(1), as memory usage does not scale with the input size.
 */
#include <iostream>
#include <string>

using namespace std;

int main()
{
  int num_operations; // Stores the total number of operations to perform.
  cin >> num_operations;

  int x_value = 0; // Represents the value of 'X', initialized to 0.
  string operation_str; // Stores the input operation, e.g., "++X".

  // Loop 'num_operations' times to process each statement.
  for (int i = 0; i < num_operations; i++)
  {
    cin >> operation_str;

    // Check if the operation is an increment.
    if (operation_str == "++X" || operation_str == "X++")
      x_value++;
    // Otherwise, the operation must be a decrement.
    else
      x_value--;
  }

  // Print the final value of X after all operations.
  cout << x_value;
}