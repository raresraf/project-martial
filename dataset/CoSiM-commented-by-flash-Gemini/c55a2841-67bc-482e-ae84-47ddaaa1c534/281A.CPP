/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of a given input string.
 *        It uses `islower` to check if the first character is lowercase and
 *        `toupper` to convert it to uppercase.
 *
 * Algorithm:
 * 1. Read a single word (string) from standard input.
 * 2. Check if the first character of the string is a lowercase letter using `islower`.
 * 3. If it is lowercase, convert the first character to uppercase using `toupper`.
 * 4. Print the modified string to standard output.
 *
 * Time Complexity: O(L) where L is the length of the input string due to string operations.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input, transformation (capitalization), and output of the string.
 * @return 0 on successful execution.
 */
int main() {
  string word;

  // Block Logic: Reads a single word (string) from standard input.
  cin >> word;

  /**
   * Block Logic: Conditionally converts the first character of the string to uppercase.
   * Uses `islower` to determine if the character is a lowercase letter, and `toupper`
   * to perform the conversion. The result of `toupper` is cast back to `char` to ensure
   * correct assignment to `word[0]`.
   *
   * Pre-condition: The string 'word' has been populated and is not empty.
   * Invariant: If the first character was lowercase, it will be uppercase; otherwise, it remains unchanged.
   */
  if (islower(word[0]))
    // Inline: Checks if the character is lowercase and converts it to uppercase.
    word[0] = char(toupper(word[0]));

  // Block Logic: Prints the potentially modified string to standard output, followed by a newline.
  cout << word << endl;

  return 0;
}

/**
ApPLe
ApPLe

konjac
Konjac

**/
