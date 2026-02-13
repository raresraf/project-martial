/**
 * @file 58A.cpp
 * @brief An incomplete and non-functional attempt to solve the "Chat room" problem (Codeforces 58A).
 * @author Tico
 * @date 3/18/17
 *
 * @details This program is intended to check if the word "hello" is a subsequence
 * in a given input string.
 *
 * @warning This code is incomplete and incorrect. It only contains logic to find
 * the first character 'h' and does not check for the rest of the subsequence.
 * It also fails to produce any output ("YES" or "NO") and will not terminate
 * correctly for typical competitive programming inputs.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  char c;

  /**
   * Block Logic: This loop reads the input stream character by character,
   * skipping whitespace.
   * @note Flaw: The loop never breaks, so it will continue until the end of the
   * input stream (e.g., EOF), which may result in a "Time Limit Exceeded" error.
   */
  while (cin >> c) {
    // The code only checks for the first character 'h' of the subsequence.
    if (c == 'h') {
        // Incomplete: Logic to find the next character 'e' should be here.
    } else {
        // If the character is not 'h', it simply continues to the next character.
      continue;
    }
  }
  
  // Incomplete: The code is missing the final check to see if the entire
  // "hello" subsequence was found and does not print "YES" or "NO".

  return 0;
}
