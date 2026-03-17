/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 71A "Way Too Long Words".
 *
 * @details
 * This program attempts to shorten words that are longer than 10 characters.
 * The intended abbreviation format is the first character, followed by the number
 * of characters between the first and last, and then the last character.
 *
 * @algorithm
 * The program reads all words into an array. It then iterates through them and
 * attempts to modify any word longer than 10 characters in-place.
 *
 * @flaw
 * This implementation is non-functional and contains multiple severe bugs:
 * 1. **Manual Integer-to-String:** It tries to manually convert the middle
 *    length into a string of digits, but does so in reverse.
 * 2. **String Reconstruction:** The logic to place these digits back into the
 *    original string is broken. It has off-by-one errors and accesses arrays
 *    out of bounds. The original string's content is corrupted unpredictably.
 * 3. **String Termination:** It attempts to shorten the `std::string` by inserting
 *    null characters (`\0`). This is not the correct way to resize a `std::string`
 *    and leads to incorrect behavior. The `std::string::resize` method should be used instead.
 *
 * As a result, the program does not produce the correct output.
 *
 * @complexity
 * - Time Complexity: O(L_total), where L_total is the sum of lengths of all words,
 *   dominated by I/O and looping.
 * - Space Complexity: O(L_total), for storing all words in an array.
 */
#include <bits/stdc++.h>

using namespace std;

int n;
string s[105];
char first;
char last;
int number;
char x[100];

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int j = 0;
  int k = 0;

  cin >> n;

  // First, read all words into an array of strings.
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }

  // Second, iterate through the stored words to process them.
  for (int i = 0; i < n; i++) {

    if (s[i].length() > 10) {
      
      first = s[i][0];
      last = s[i][s[i].length() - 1];
      number = s[i].length() - 2;

      // --- BUGGY LOGIC BLOCK START ---

      /**
       * @brief FLAW 1: Buggy manual conversion of integer to character array.
       * This loop extracts digits in reverse order. For number 12, `x` becomes "21".
       */
      j = 0;
      k = 0;
      while (number != 0) {
        x[j++] = '0' + number % 10;
        number /= 10;
      }

      /**
       * @brief FLAW 2: Buggy string reconstruction.
       * This loop attempts to copy the reversed digits back into the original string `s[i]`.
       * The loop condition `j >= 0` and pre-decrement `--j` will cause an out-of-bounds
       * read from `x` on the final iteration (when j is -1). The logic corrupts the string.
       */
      while (j >= 0) {
        s[i][++k] = x[--j];
      }
      s[i][k++] = last;

      /**
       * @brief FLAW 3: Incorrectly terminating a std::string.
       * This does not resize the string. It inserts null characters, which may
       * cause `cout` to stop printing early, but the string's length remains the same.
       * The correct method would be `s[i].resize(k)`.
       */
      for (int p = k; p < s[i].length(); p++)
        s[i][p] = '\0';

      // --- BUGGY LOGIC BLOCK END ---
      j = 0;
      k = 0;
    }
  }

  // Print the (likely corrupted) results.
  for (int i = 0; i < n; i++) {
    cout << s[i] << endl;
  }

  return 0;
}
