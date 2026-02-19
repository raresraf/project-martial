/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem determining user interaction based on username character diversity.
 * @details This program analyzes a username to count its unique characters. The decision to
 * "CHAT WITH HER!" or "IGNORE HIM!" is based on whether the count of unique characters is even or odd.
 *
 * @algorithm This implementation uses an in-place string modification approach to find unique characters.
 * It employs a nested loop to iterate through the string. For each character, it scans the
 * remainder of the string for duplicates. Any duplicates found are removed directly from the string
 * using `string::erase`.
 *
 * @time_complexity O(N^2), where N is the length of the input string. The nested loop structure for
 * finding duplicates combined with the linear-time `string::erase` operation leads to a quadratic runtime.
 * @space_complexity O(1) extra space, as the duplicate removal is performed in-place on the
 * input string.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string x;
  cin >> x;

  /**
   * Block Logic: This nested loop performs in-place removal of duplicate characters.
   * Outer loop selects a character to check against.
   * Inner loop scans the rest of the string for duplicates of the selected character.
   * Invariant: At the end of iteration `i` of the outer loop, the character `x[i]` is unique
   *            within the prefix `x[0...i]`.
   */
  for (int i = 0; i < x.size(); i++) {
    for (int j = i + 1; j < x.size(); j++) {
      if (x[i] == x[j]) {
        // A duplicate is found. Remove it from the string.
        x.erase(j, 1);
        // Inline: Decrement j to correctly handle the next character after the shift caused by erase.
        j--;
      }
    }
  }
  /**
   * Block Logic: Determine the output based on the parity of the number of unique characters.
   * Pre-condition: The string 'x' now contains only the unique characters from the original input.
   */
  if (x.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
  return 0;
}
