/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove so that no two adjacent stones are of the same color.
 *
 * This program solves the "Stones on the Table" problem, which involves finding the number of
 * adjacent identical characters in a string. This count represents the minimum number of stones
 * that need to be removed to satisfy the condition.
 *
 * @see https://codeforces.com/problemset/problem/266/A
 *
 * Algorithm:
 * A single pass is made through the sequence of stones (represented as a string). The color of each
 * stone is compared with the color of the next one. If they are the same, a counter is incremented.
 *
 * Time Complexity: O(N), where N is the number of stones, as we iterate through the sequence once.
 * Space Complexity: O(N) to store the string representing the stones.
 */
#include <iostream>
#include <string>

using namespace std;
int main() {
  // n: the number of stones on the table.
  int n;
  std::cin >> n;
  
  // s: a string representing the colors of the stones.
  std::string s;
  std::cin >> s;
  
  // count: stores the number of stones to be removed.
  int count = 0;
  
  /**
   * @brief Iterates through the stones to find adjacent pairs of the same color.
   *
   * Pre-condition: The string 's' contains the sequence of stone colors.
   * Invariant: 'count' holds the number of removals needed for the first 'i' stones.
   */
  for (int i = 0; i < n - 1; i++) {
    // If the current stone has the same color as the next one, increment the count.
    if (s[i] == s[i + 1]) {
      count += 1;
    }
  }
  
  // Output the total number of stones to remove.
  std::cout << count;
  return 0;
}
