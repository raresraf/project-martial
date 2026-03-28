/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 *
 * This program simulates a series of increment and decrement operations on a
 * variable, `X`, initialized to zero. It calculates and prints the final value of
 * `X` after all operations are performed.
 *
 * @algorithm The program iterates `n` times, once for each operation. In each
 * iteration, it reads the operation as a string. It then uses `string::find`
 * to check for the presence of the substring "++". If found, the variable `X`
 * is incremented. Otherwise, it's assumed the operation is a decrement, and `X`
 * is decremented.
 *
 * @time_complexity O(N), where N is the number of operations.
 * @space_complexity O(1), as extra space is minimal.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  int X = 0;
  cin >> n;
  /**
   * @block_logic This loop processes each of the `n` given operations.
   */
  for (int i = n; i > 0; i--) {
    string op;
    cin >> op;
    /**
     * @block_logic Checks if the operation is an increment by searching for "++".
     * If "++" is found, `op.find` will not return `string::npos`. Otherwise,
     * the operation is assumed to be a decrement.
     */
    if (op.find("++") != string::npos) {
      X++;
    } else {
      X--;
    }
  }
  cout << X << endl;
}