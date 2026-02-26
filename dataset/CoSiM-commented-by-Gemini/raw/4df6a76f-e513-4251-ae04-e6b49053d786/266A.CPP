/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces.
 * @details The problem asks for the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color. This
 * is equivalent to counting the number of adjacent pairs of identical stones.
 *
 * @algorithm
 * 1. Read the number of stones and the string representing their colors.
 * 2. Iterate through the string from the first character up to the
 *    second-to-last character.
 * 3. In each step, compare the current character (stone) `s[i]` with the next
 *    one `s[i+1]`.
 * 4. If they are identical, increment a counter. This counter represents a stone
 *    that needs to be removed.
 * 5. After the loop finishes, the value of the counter is the minimum number
 *    of stones to remove.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stones, due to the single
 * pass through the string.
 * Space Complexity: O(N) to store the input string of stone colors.
 */

/*
 *Link: https://codeforces.com/problemset/problem/266/A
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string stones;
  int n; // The number of stones (redundant, as string length can be used).
  int stones_to_remove = 0; // Counter for the result.

  cin >> n >> stones;

  /**
   * @block
   * @brief Count adjacent identical stones.
   * @details The loop iterates up to the second-to-last element to allow for
   * comparison between `stones[i]` and `stones[i+1]`.
   */
  for (int i = 0; i < stones.length() - 1; i++) {
    // If a stone has the same color as its right neighbor, one must be removed.
    if (stones[i] == stones[i + 1])
      stones_to_remove++;
  }
  
  // Print the total count of stones that need to be removed.
  cout << stones_to_remove << endl;
}
