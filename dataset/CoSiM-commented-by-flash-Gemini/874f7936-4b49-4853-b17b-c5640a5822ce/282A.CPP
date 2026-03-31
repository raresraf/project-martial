/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces (Problem 282A).
 * This program calculates the final value of a variable 'X' after a series of
 * increment (++) or decrement (--) operations.
 *
 * Algorithm: Counter simulation. The program iterates through 'N' operations,
 * parsing each operation string to determine if 'X' should be incremented or
 * decremented, and updates a counter accordingly.
 * Time Complexity: O(N * L), where N is the number of operations and L is the
 *                  fixed length of the operation string (typically 3).
 *                  Effectively O(N) due to fixed string length.
 * Space Complexity: O(L) for storing the operation string, or O(1) if input is streamed.
 */

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm).
using namespace std;
int main() {
  /**
   * @brief Main function to simulate Bit++ operations and calculate the final value.
   * Reads the number of operations 'n', then processes 'n' operation strings.
   * Increments or decrements a counter based on the operation and prints the final count.
   */
  int n, count = 0; // @brief n: number of operations; count: current value of X, initialized to 0.
  cin >> n;
  
  // Block Logic: Iterate through each operation.
  // Pre-condition: 'n' operations are expected.
  // Invariant: 'count' accurately reflects the value of X after each operation.
  for (int i = 0; i < n; i++) {
    string x; // @brief String representing the current operation (e.g., "++X", "X--").
    cin >> x;
    
    // Block Logic: Determine the operation and update the counter.
    // Checks for "++" at the beginning or middle of the string to increment.
    if (x[0] == '+' && x[1] == '+')
      count++; // Inline: Increment count for "++X" or similar.
    // Checks for "--" at the beginning or middle of the string to decrement.
    else if (x[0] == '-' || x[1] == '-') // Inline: More robust check, covers "X--" as well.
      count--; // Inline: Decrement count for "--X" or similar.
    // The following two 'else if' conditions are redundant given the previous checks,
    // as any valid "++X", "X++", "--X", "X--" operation would be caught by the first two.
    // However, they are part of the original code, so they are documented as-is.
    else if (x[1] == '+' && x[2] == '+')
      count++; // Inline: Increment count for "X++".
    else if (x[1] == '-' && x[2] == '-')
      count--; // Inline: Decrement count for "X--".
  }
  // Block Logic: Print the final value of X.
  cout << count << endl;
  return 0; // @brief Indicates successful program termination.
}