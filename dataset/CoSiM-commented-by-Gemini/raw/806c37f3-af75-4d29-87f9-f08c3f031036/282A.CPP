/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem involving simple arithmetic operations.
 * @details This program simulates a series of increment and decrement operations on a variable.
 * It reads an integer 'x' representing the number of operations, and then for each
 * operation, it reads a string. If the string contains a '+', a counter 'k' is
 * incremented. If it contains a '-', 'k' is decremented. The final value of 'k' is printed.
 *
 * @algorithm Simple loop-based simulation.
 * @complexity
 * Time complexity: O(N*M), where N is the number of operations and M is the length of the operation string.
 * However, since the inner loop breaks after finding the first '+' or '-', the effective complexity is closer to O(N).
 * Space complexity: O(L), where L is the maximum length of an operation string.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to execute the solution.
 */
int main() {

  /**
   * @param x The number of operations to perform.
   * @param k The variable that is incremented or decremented.
   */
  int x;

  cin >> x;

  int k = 0;

  /**
   * @brief This loop processes each of the 'x' operations.
   */
  for (int i = 0; i < x; i++) {

    string s;

    cin >> s;

    /**
     * @brief This loop checks for a '+' or '-' in the operation string.
     */
    for (int i = 0; i < s.length(); i++) {

      if (s[i] == '+') {

        k++;

        break;
      }

      if (s[i] == '-') {

        k--;

        break;
      }
    }
  }

  cout << k;
}
