/**
 * @67ea5045-79ca-493d-9505-25291cf1f928/282A.CPP
 * @brief This program solves Codeforces problem 282A, "Bit++".
 * It simulates a simple counter variable that is modified by a series of
 * input command strings. Each command represents either an increment
 * (e.g., "X++", "++X") or a decrement (e.g., "X--", "--X") operation.
 * The program reads the number of commands and then processes each command
 * to determine the final value of the counter. This solution checks the
 * middle character of the 3-character string ("+" or "-") to determine the operation.
 *
 * Algorithm: Iterative processing of string commands to update an integer counter.
 * String character access is used to identify increment/decrement operations.
 * Time Complexity: O(N) where N is the number of commands.
 * Space Complexity: O(L) for storing the command string (constant, typically 3).
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to process a series of increment/decrement commands.
 * Reads the number of operations, then each operation string, updates a counter,
 * and prints the final counter value.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares 'n', representing the total number of operations.
  cin >> n; // Reads 'n' from standard input.
  
  string str; // Declares a string variable to store each operation command.
  int x = 0; // Initializes 'x' (the counter variable) to 0.
  
  /**
   * Block Logic: Processes 'n' operation commands.
   * For each command, it checks if the middle character is '+' or '-' to
   * determine if it's an increment or decrement operation, and updates `x` accordingly.
   * Invariant: `x` holds the cumulative result of all operations.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    cin >> str; // Reads an operation command string (e.g., "X++", "++X", "X--", "--X").
    
    // Block Logic: Determines if the command is an increment or decrement based on the middle character.
    // The problem statement implies commands are 3 characters long, so index 1 is the middle character.
    if (str[1] == '+') { // If the middle character is '+', it's an increment.
      x++; // Increments 'x'.
    } else if (str[1] == '-') { // If the middle character is '-', it's a decrement.
      x--; // Decrements 'x'.
    }
  }
  cout << x; // Prints the final value of the counter.
  return 0; // Indicates successful program termination.
}
