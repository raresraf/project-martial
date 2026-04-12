/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem from Codeforces.
 *
 * This program reads a username string, counts the number of unique characters
 * in it, and then determines whether this count is even or odd. Based on this
 * parity, it prints "CHAT WITH HER!" or "IGNORE HIM!".
 *
 * The core idea is to find the number of distinct characters, which directly
 * relates to the problem's logic of determining gender-based chat responses.
 *
 * Algorithm:
 * 1. Read the input string (username).
 * 2. Sort the string to bring identical characters together.
 * 3. Use `unique` to remove adjacent duplicate characters, effectively counting
 *    unique characters.
 * 4. Determine if the length of the string with unique characters is even or odd.
 * 5. Print the corresponding message.
 *
 * Time Complexity: O(N log N) due to sorting, where N is the length of the string.
 * Space Complexity: O(N) for storing the string.
 */
// 4022081   Jul 4, 2013 8:27:39 PM	fuwutu	 236A - Boy or Girl	 GNU
// C++0x	Accepted	15 ms	0 KB
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function: entry point of the program.
 *
 * Orchestrates the reading of input, processing the string to find unique characters,
 * and printing the final output based on the parity of the unique character count.
 * @return 0 on successful execution.
 */
int main() {
  string s;
  cin >> s;
  /**
   * @brief Sorts the input string.
   *
   * Pre-condition: `s` contains the input username.
   * Post-condition: Characters in `s` are arranged in non-decreasing order.
   * This operation is crucial for `std::unique` to work correctly, as `unique`
   * only operates on contiguous duplicate elements.
   */
  sort(s.begin(), s.end());
  /**
   * @brief Removes duplicate characters from the sorted string.
   *
   * Pre-condition: `s` is sorted.
   * `std::unique` rearranges the string such that unique elements are at the beginning
   * and returns an iterator to the new end of the unique range.
   * `s.erase` then removes the elements from this new end to the original end of the string.
   * Invariant: After this operation, `s` contains only unique characters from the original input.
   */
  s.erase(unique(s.begin(), s.end()), s.end());
  /**
   * @brief Determines output based on the parity of unique characters.
   *
   * Pre-condition: `s` contains only unique characters from the original input.
   * If the number of unique characters (length of `s`) is even, it suggests a "boy" context,
   * leading to the "CHAT WITH HER!" output. Otherwise, for an odd count ("girl" context),
   * it prints "IGNORE HIM!". This implements the problem's core decision logic.
   */
  if (s.length() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}
