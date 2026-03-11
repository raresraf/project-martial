/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid consecutive identical colors.
 *
 * This program solves a classic problem where stones of different colors are arranged
 * in a row. The goal is to find the minimum number of stones to remove so that no
 * two adjacent stones have the same color. This is equivalent to counting the
 * number of pairs of adjacent, identical characters in a string.
 *
 * @algorithm The program uses a single pass through the string to count adjacent
 * identical characters. It iterates from the beginning of the string up to the
 * second-to-last character and compares each character with its next neighbor.
 *
 * @time_complexity O(N), where N is the length of the string (`test`), because it
 * iterates through the string once.
 * @space_complexity O(N) to store the input string `word`.
 */
#include <iostream>

using namespace std;

int main() {
  string word; // Represents the sequence of stone colors.
  int test; // Represents the number of stones, which is the length of the string.

  // Read the number of stones and the sequence of colors.
  cin >> test >> word;

  auto count = 0; // Counter for the number of stones to remove.

  /**
   * @brief Loop to iterate through the stones and count adjacent identical ones.
   * @invariant `count` holds the number of removals needed for the substring `word[0...i]`.
   * @pre The loop runs from the first character to the second-to-last character.
   */
  for (auto i = 0; i < test - 1; i++) {
    // If a character is the same as the one immediately following it,
    // one of them needs to be "removed".
    if (word[i] == word[i + 1])
      count++;
  }

  // Print the total number of stones to remove.
  cout << count << endl;

  return 0;
}
