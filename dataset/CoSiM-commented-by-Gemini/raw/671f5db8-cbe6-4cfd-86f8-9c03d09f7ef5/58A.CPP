#include <iostream>
#include <string>

using namespace std;

/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem from Codeforces (Problem 58A).
 * @details This program checks if a given string contains "hello" as a subsequence.
 * A subsequence is a sequence that can be derived from another sequence by deleting
 * some or no elements without changing the order of the remaining elements.
 *
 * Algorithm:
 * The program uses a state machine approach, managed by a `flag` variable, to find
 * the subsequence "hello". It iterates through the input string character by character.
 * The `flag` variable tracks which character of "hello" is currently being sought.
 * - flag 0: looking for 'h'
 * - flag 1: found 'h', looking for 'e'
 * - flag 2: found "he", looking for first 'l'
 * - flag 3: found "hel", looking for second 'l'
 * - flag 4: found "hell", looking for 'o'
 * When the expected character is found, the flag is incremented. If the flag reaches 5,
 * it means the full "hello" subsequence has been found, and the program prints "YES"
 * and exits. If the loop completes and the flag is less than 5, it prints "NO".
 *
 * Time Complexity: O(N), where N is the length of the input string, due to the single pass.
 * Space Complexity: O(N) to store the input string.
 */
int main() {
  string input;
  getline(cin, input);

  int n;
  n = input.length();

  // `flag` acts as a state machine counter for matching "hello".
  int flag = 0;

  /**
   * @brief Iterates through the input string to find the "hello" subsequence.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief State machine logic.
     * Checks if the current character matches the character expected for the current state.
     */
    if ((flag == 0 && input.at(i) == 'h') ||
        (flag == 1 && input.at(i) == 'e') ||
        ((flag == 2 || flag == 3) && input.at(i) == 'l') ||
        (flag == 4 && input.at(i) == 'o')) {
      // If a character is matched, advance to the next state.
      flag++;
    }

    // If flag reaches 5, all characters of "hello" have been found in order.
    if (flag == 5) {
      cout << "YES";
      return 0;
    }
  }

  // If the loop finishes and flag is not 5, the subsequence was not found.
  cout << "NO";
  return 0;
}
