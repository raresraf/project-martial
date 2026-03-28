/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 *
 * This program simulates a series of increment (X++, ++X) and decrement (X--, --X)
 * operations on a variable, `x`, initialized to zero. It calculates and prints
 * the final value of `x` after all operations.
 *
 * @algorithm The program iterates `n` times, where `n` is the number of operations.
 * In each iteration, it reads the entire operation as a string. It then uses
 * explicit string comparison to check if the command is one of the four valid
 * increment or decrement operations and updates a counter variable accordingly.
 *
 * @time_complexity O(N), where N is the number of operations. String comparison
 * takes constant time as the strings are of fixed length 3.
 * @space_complexity O(1), aside from the storage for the input strings.
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  int x = 0;
  cin >> n;

  /**
   * @block_logic This loop processes each of the `n` given operations.
   */
  for (int i = 0; i < n; i++) {
    string command;
    cin >> command;
    /**
     * @block_logic This block explicitly compares the input string against the four
     * possible operations and updates the value of `x` accordingly.
     */
    if (command.compare("X++") == 0 || command.compare("++X") == 0) {
      x = x + 1;
    }
    if (command.compare("X--") == 0 || command.compare("--X") == 0) {
      x = x - 1;
    }
  }
  cout << x << "\n";
  return 0;
}