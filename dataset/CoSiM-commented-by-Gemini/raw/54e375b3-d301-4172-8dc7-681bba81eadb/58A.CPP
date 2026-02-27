/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces.
 * @details The problem asks to determine if a given string contains "hello" as a
 * subsequence. A subsequence is formed by deleting zero or more characters
 * from a string without changing the order of the remaining characters.
 *
 * @algorithm
 * A two-pointer approach is used to efficiently check for the subsequence.
 * 1. Define the target subsequence, "hello".
 * 2. Use a pointer `i` for the input string and a pointer `index` for the target
 *    subsequence "hello".
 * 3. Iterate through the input string with `i`.
 * 4. If the character at the current position in the input string matches the
 *    character at the current `index` of "hello", advance the `index` pointer.
 * 5. This effectively "finds" the characters of "hello" in the correct order,
 *    skipping any non-matching characters in the input string.
 * 6. After the loop, if the `index` pointer has reached the end of "hello",
 *    it means the entire subsequence was found.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, due to a
 * single pass.
 * Space Complexity: O(L) to store the input string.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  string line; // The input string.
  cin >> line;
  
  string temp = "hello"; // The target subsequence.
  int index = 0;         // A pointer to the current character in the target subsequence.

  /**
   * @block
   * @brief Iterates through the input string to find the subsequence.
   * The loop uses 'i' to traverse the input string `line` and `index` to
   * traverse the target string `temp`.
   */
  for (auto i = 0; i < line.length() && index < temp.length(); i++) {
    // If a character matches the current target character, advance the target pointer.
    if (line[i] == temp[index]) {
      index++;
    }
  }
  
  /**
   * @brief Check if the entire subsequence was found.
   * If `index` equals the length of "hello", then all its characters were found
   * in the correct order.
   */
  if (index == temp.length())
    cout << "YES";
  else
    cout << "NO";
    
  return 0;
}
