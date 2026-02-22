/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A: Stones on the Table.
 * @details This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones are of the same color. The input consists of the
 * total number of stones 'n' and a string 's' representing the colors of the stones.
 * The solution counts the number of adjacent pairs of identical colored stones.
 * @algorithm Linear scan for adjacent identical elements.
 * Time Complexity: $O(N)$ where N is the number of stones (length of the string).
 * Space Complexity: $O(N)$ to store the string.
 */
#include <iostream>

using namespace std;

/**

 * @file 266A.CPP

 * @brief Solution for Codeforces Problem 266A: Stones on the Table.

 * @details This program calculates the minimum number of stones to remove from a row

 * such that no two adjacent stones are of the same color. The input consists of the

 * total number of stones 'n' and a string 's' representing the colors of the stones.

 * The solution counts the number of adjacent pairs of identical colored stones.

 * @algorithm Linear scan for adjacent identical elements.

 * Time Complexity: $O(N)$ where N is the number of stones (length of the string).

 * Space Complexity: $O(N)$ to store the string.

 */

#include <iostream> // Required for input/output operations (std::cin, std::cout).



using namespace std; // Uses the standard namespace.



int main() {

  /**

   * @brief Main function: orchestrates the stone removal calculation logic.

   */

  string s; // @var s: The input string representing the colors of the stones.

  int i,    // @var i: Loop counter.

      n;    // @var n: The total number of stones.



  cin >> n >> s; // Reads the total number of stones 'n' and the string 's'.

  int c = 0;   // @var c: Counter for the number of adjacent identical stone pairs.



  /**

   * @brief Iterates through the stone string to count adjacent identical pairs.

   * @block_logic Compares each stone with its immediate successor.

   * @invariant `c` accumulates the count of pairs where `s[i]` is equal to `s[i+1]`.

   */

  for (i = 0; i < n - 1; i++) {

    if (s[i] == s[i + 1]) // Checks if the current stone has the same color as the next one.

      c++;                 // Increments the counter if they are identical.

  }



  // Functional Utility: Prints the total number of adjacent identical stone pairs (minimum removals).

  cout << c << endl;

  return 0; // Indicates successful execution.

}
