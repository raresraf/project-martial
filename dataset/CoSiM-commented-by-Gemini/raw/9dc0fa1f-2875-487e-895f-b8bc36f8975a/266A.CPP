/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid consecutive identical colors.
 *
 * This program solves a problem where stones of different colors are arranged
 * in a row. The goal is to find the minimum number of stones to remove so that no
 * two adjacent stones have the same color. This is equivalent to counting the
 * number of pairs of adjacent, identical characters in a string.
 *
 * @algorithm The program uses a single pass through the string to count adjacent
 * identical characters. It iterates through the string and compares each character
 * with its next neighbor.
 *
 * @note The loop condition `i < len` combined with the access `s[i + 1]`
 * causes an out-of-bounds read on the final iteration (`i = len - 1`), which is
 * undefined behavior. The loop should run until `i < len - 1`.
 *
 * @time_complexity O(N), where N is the length of the string, as it
 * iterates through the string once.
 * @space_complexity O(N) to store the input string `s`.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int n; // The number of stones (length of the string).
  cin >> n >> s;
  int len = s.length(); // Redundant, as `n` already holds the length.
  int count = 0; // Counter for the number of stones to remove.

  /**
   * @brief Loop to iterate through the stones and count adjacent identical ones.
   * @warning This loop condition `i < len` is incorrect. On the last iteration,
   * `i` will be `len - 1`, and `s[i + 1]` (i.e., `s[len]`) will be an
   * out-of-bounds access. The loop should be `for (int i = 0; i < len - 1; i++)`.
   */
  for (int i = 0; i < len; i++) {
    // If a stone's color is the same as the one immediately following it,
    // one of them needs to be "removed".
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  
  // Print the total number of stones to remove.
  cout << count;
  return 0;
}