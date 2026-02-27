/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces.
 * @details The problem asks for the minimum number of stones to remove from a
 * row of colored stones such that no two adjacent stones have the same color.
 *
 * @algorithm
 * To minimize removals, we should only remove a stone if it has the same color
 * as the one next to it. The algorithm iterates through the row of stones and
 * counts how many pairs of adjacent stones are identical. This count represents
 * the minimum number of stones that must be removed.
 * 1. Read the number of stones, `no`.
 * 2. Read the string representing the colors of the stones.
 * 3. Initialize a counter `output` to 0.
 * 4. Iterate through the stones from the second stone to the end (index 1 to n-1).
 * 5. In each iteration, compare the color of the current stone with the color of
 *    the previous stone.
 * 6. If they are the same, increment the `output` counter.
 * 7. After the loop, `output` holds the total minimum number of stones to remove.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of stones, due to the single pass.
 * Space Complexity: O(n) to store the string of colors.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int no; // The number of stones.
  // 'output' will count the minimum number of stones to remove.
  int output = 0;
  // 'lastcolor' and 'color' will be used for comparing adjacent stones.
  char lastcolor, color;
  // 'temp' is declared but never used.
  char temp;
  string colors; // The string representing the colors of the stones.
  
  cin >> no;
  cin >> colors;

  // The logic only applies if there is more than one stone.
  if (no > 1) {
    // This line is redundant and has no effect on the logic.
    // color = colors.at(1);
    
    /**
     * @block
     * @brief Iterate through the stones to find adjacent pairs of the same color.
     */
    for (int i = 1; i < no; i++) {
      color = colors.at(i);
      lastcolor = colors.at(i - 1);
      
      // If an adjacent pair has the same color, one must be removed.
      if (lastcolor == color) {
        output++;
      }
    }
  }

  // Print the total number of stones to remove.
  cout << output;
  return 0;
}
