/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces.
 * @details The program determines a user's gender based on the number of
 * distinct characters in their name. If the count of distinct characters
 * is even, the user is female ("CHAT WITH HER!"). If it's odd, the user
 * is male ("IGNORE HIM!").
 *
 * Algorithm:
 * 1. Read the user's name as a string.
 * 2. Sort the string alphabetically. This groups all identical characters together.
 * 3. Iterate through the sorted string and count the number of "transitions"
 *    where a character is different from the one immediately following it. This
 *    count equals the number of distinct characters.
 * 4. Check if the final count is even or odd.
 * 5. Print "CHAT WITH HER!" for an even count, and "IGNORE HIM!" for an odd count.
 *
 * Time Complexity: O(L log L), dominated by the sorting of the string, where L
 * is the length of the string.
 * Space Complexity: O(L), to store the string.
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  string str;
  cin >> str;

  int counter = 0;

  sort(str.begin(), str.end());

  /**
   * Block Logic: Count the number of distinct characters by checking for
   * character changes in the sorted string.
   * NOTE: The loop condition `i < str.length()` can lead to an out-of-bounds read
   * with `str[i + 1]` on the last iteration. It often works by chance because
   * `str[str.length()]` is a null terminator, which is different from the last
   * character of a typical string. A safer loop would be `i < str.length() - 1`.
   */
  for (unsigned int i = 0; i < str.length(); i++) {
    if (str[i] != str[i + 1]) {
      counter++;
    }
  }
  /**
   * Block Logic: Check the parity of the distinct character count to determine the output.
   */
  if (counter % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}