/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem: "Bit++".
 * Processes a sequence of operations (increment or decrement) on a single variable 'x'.
 * Algorithm: Direct simulation of operations.
 * Time Complexity: O(N) where N is the number of operations.
 * Space Complexity: O(1)
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to solve the Bit++ problem.
 * Reads the number of operations 'n' and then processes 'n' operations.
 * Each operation is either "++X", "X++", "--X", or "X--".
 * The program simulates these operations on an integer variable 'x' and prints its final value.
 */
int main() {
  int n, x = 0;
  cin >> n;
  string opr;
  /**
   * Block Logic: Iterates through each operation and updates the variable 'x'.
   * Pre-condition: 'n' operations are pending, 'x' is initialized to 0.
   * Invariant: At the start of each iteration, 'x' holds the result of previous operations.
   */
  for (int i = 0; i < n; i++) {
    cin >> opr;
    /**
     * Block Logic: Checks if the operation is an increment.
     * This covers "++X" and "X++" as both have '+' at index 1.
     */
    if (opr.at(1) == '+')
      x++;
    /**
     * Block Logic: Checks if the operation is a decrement.
     * This covers "--X" and "X--" as both have '-' at index 1.
     */
    if (opr.at(1) == '-')
      x--;
  }
  cout << x << endl;

  return 0;
}

