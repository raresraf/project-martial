/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid consecutive identical colors.
 *
 * This program solves a classic problem where stones of different colors are arranged
 * in a row. The goal is to find the minimum number of stones to remove so that no
 * two adjacent stones have the same color. This is equivalent to counting the
 * number of pairs of adjacent, identical characters in a string. This version uses
 * a C-style character array.
 *
 * @algorithm The program uses a single pass through the array to count adjacent
 * identical characters. It iterates from the beginning of the array up to the
 * second-to-last character and compares each character with its next neighbor.
 *
 * @note The use of a variable-length array (`char s[n]`) is a non-standard C++
 * extension. While supported by some compilers like GCC, it is not part of the
 * C++ standard and may not be portable.
 *
 * @time_complexity O(N), where N is the number of stones, because it
 * iterates through the character array once.
 * @space_complexity O(N) to store the input sequence of stone colors.
 */
#include <iostream>

using namespace std;

int main() {
  int n; // Represents the number of stones.
  int c = 0; // Counter for the number of stones to remove.

  // Read the number of stones.
  cin >> n;

  // Declare a C-style character array to hold the sequence of colors.
  // Note: This is a Variable Length Array (VLA), a non-standard C++ extension.
  char s[n];

  // Read the sequence of stone colors.
  cin >> s;

  /**
   * @brief Loop to iterate through the stones and count adjacent identical ones.
   * @invariant `c` holds the number of removals needed for the subarray `s[0...i]`.
   * @pre The loop runs from the first stone to the second-to-last stone.
   */
  for (int i = 0; i < n - 1; i++) {
    // If a stone's color is the same as the one immediately following it,
    // one of them needs to be "removed".
    if (s[i] == s[i + 1]) {
      c++;
    }
  }

  // Print the total number of stones to remove.
  cout << c << endl;

  return 0;
}
