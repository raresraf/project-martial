/**
 * @88e9dae6-63b2-4402-9711-6f8b3d5b8a5a/282A.CPP
 * @brief Solution for a competitive programming problem involving simple arithmetic operations.
 *
 * This program simulates a variable 'x' undergoing a series of increment or decrement operations.
 * The operations are provided as strings, and the program determines the final value of 'x'
 * starting from an initial value of 0.
 *
 * Algorithm:
 * 1. Initialize an integer variable 'x' to 0.
 * 2. Read an integer 'n', representing the total number of operations.
 * 3. Loop 'n' times, processing one operation in each iteration:
 *    a. Read an operation string 's'.
 *    b. Examine the character at the second position (index 1) of the string.
 *    c. If this character is '+', increment 'x'.
 *    d. Otherwise (implying it's '-'), decrement 'x'.
 * 4. After all operations are processed, print the final value of 'x'.
 *
 * Time Complexity: O(N * L) where N is the number of operations and L is the length of each operation string (constant here).
 *                  Effectively O(N) as string length check is constant time.
 * Space Complexity: O(1) as only a few variables and a temporary string are used.
 */

#include <iostream> // Functional Utility: Provides standard input/output streams (std::cin, std::cout).
using namespace std;

int main() {
  int n, x = 0;
  // Functional Utility: Reads the total number of operations 'n'.
  cin >> n;

  // Block Logic: Loops 'n' times to process each operation.
  // Pre-condition: 'n' is a non-negative integer.
  // Invariant: 'x' is updated based on 'n' string operations.
  while (n--) {
    string s;
    // Functional Utility: Reads the operation string (e.g., "++X", "X--", "--X").
    cin >> s;

    // Conditional Logic: Checks the character at index 1 to determine if it's an increment operation.
    if (s.at(1) == '+')
      x++; // Functional Utility: Increments 'x'.
    else
      x--; // Functional Utility: Decrements 'x'.
  }

  // Functional Utility: Prints the final calculated value of 'x', followed by a newline.
  cout << x << endl;
  return 0; // Functional Utility: Indicates successful program execution.
}
