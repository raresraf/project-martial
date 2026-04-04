/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem on platforms like Codeforces.
 * @details This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color. This is equivalent to counting
 * the number of adjacent identical characters in a string.
 *
 * Algorithm:
 * 1. Read the number of stones, n.
 * 2. Read the sequence of stone colors as a string.
 * 3. Initialize a counter 'num' for stones to be removed to 0.
 * 4. Initialize 'temp' with the color of the first stone.
 * 5. Iterate from the second stone to the end of the row:
 *    a. If the current stone's color is the same as 'temp', increment 'num'.
 *    b. Otherwise, update 'temp' to the current stone's color.
 * 6. Print the total count 'num'.
 *
 * Time Complexity: O(N), where N is the number of stones.
 * Space Complexity: O(N) to store the string of stone colors.
 */
#include <cstring>
#include <iostream>

int main(int argc, char const *argv[]) {
  int i, n, num = 0;
  char str[100], temp;
  std::cin >> n;
  std::cin >> str;
  temp = str[0];
  /**
   * @brief Loop through the stones to find adjacent pairs of the same color.
   * Pre-condition: 'n' is the number of stones, 'str' contains their colors.
   * 'temp' holds the color of the previous stone considered, initialized to str[0].
   * Invariant: 'num' counts the number of stones to remove up to the current position 'i'.
   */
  for (i = 1; i < n; i++) {
    // If the current stone has the same color as the previous one, it needs to be removed.
    if (str[i] == temp)
      num++;
    else
      // Otherwise, the current stone becomes the new reference for the next comparison.
      temp = str[i];
  }
  // Output the total number of stones to be removed.
  std::cout << num << '\n';
  return 0;
}
