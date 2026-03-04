/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A - "Stones on the Table".
 *
 * This program calculates the minimum number of stones to remove from a row
 * on a table so that no two adjacent stones have the same color. The colors
 * are represented by characters in a string. The problem is equivalent to
 * counting the number of adjacent identical characters.
 *
 * @algorithm Linear scan and counting.
 * @time_complexity O(N), where N is the number of stones (the length of the
 *                  string), as it iterates through the string once.
 * @space_complexity O(N) to store the input string.
 */
#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Main function to count adjacent identical characters.
 */
int main() {
  int n, count = 0;
  // Read the number of stones.
  cin >> n;
  string str;
  // Read the string representing the colors of the stones.
  cin >> str;
  /**
   * @brief Counts the number of adjacent identical stones.
   * This loop iterates through the string, starting from the second character,
   * and compares each character with the one before it. If they are the same,
   * it means a stone needs to be removed, so the counter is incremented.
   */
  for (int i = 1; i < n; i++) {
    if (str[i] == str[i - 1]) {
      count++;
    }
  }
  // Print the total number of stones to remove.
  std::cout << count;
  return 0;
}
