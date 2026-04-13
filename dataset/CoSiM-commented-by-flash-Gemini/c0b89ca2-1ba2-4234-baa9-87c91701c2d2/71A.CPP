/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic. For each input word,
 *        if its length exceeds 10 characters, it is abbreviated to its first
 *        letter, the count of intermediate characters, and its last letter
 *        (e.g., "localization" becomes "l10n"). Otherwise, the word remains unchanged.
 * Algorithm: String length-based abbreviation.
 * Time Complexity: O(N * L) where N is the number of words and L is the maximum
 *                  length of a word, as each word is processed once.
 * Space Complexity: O(N * L) for storing the input and processed words.
 */
#include <bits/stdc++.h>

using namespace std;

// n: number of words to process.
// s[105]: array of strings to store words (max 105 words).
// first: stores the first character of a word.
// last: stores the last character of a word.
// number: stores the count of intermediate characters for abbreviation.
// x[100]: temporary character array to store digits of 'number' in reverse.
int n;
string s[105];
char first;
char last;
int number;
char x[100];

int main() {
  int j = 0; // Temporary counter for array 'x'.
  int k = 0; // Temporary counter for modifying string 's[i]'.

  // Reads the number of words to be processed.
  cin >> n;

  /**
   * Block Logic: Reads 'n' words from standard input into the string array 's'.
   * Invariant: After each iteration, one word is successfully read into 's[i]'.
   */
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }

  /**
   * Block Logic: Iterates through each word and applies the abbreviation logic if necessary.
   * Invariant: After each iteration, the word 's[i]' is either abbreviated or remains unchanged.
   */
  for (int i = 0; i < n; i++) {
    // Checks if the current word's length is greater than 10.
    if (s[i].length() > 10) {
      // cout<<x[i]; // Commented out debug print.
      first = s[i][0]; // Stores the first character.
      // cout<<first; // Commented out debug print.
      last = s[i][s[i].length() - 1]; // Stores the last character.
      // cout<<last; // Commented out debug print.
      // Calculates the number of intermediate characters (total length - first - last).
      number = s[i].length() - 2;
      // cout<<number; // Commented out debug print.

      j = 0; // Resets temporary counter for 'x'.
      k = 0; // Resets temporary counter for string modification.
      /**
       * Block Logic: Converts the 'number' (count of intermediate characters) into
       *              its string representation, storing digits in reverse order in 'x'.
       * Invariant: 'x' contains the digits of 'number' in reverse, and 'j' is the count of digits.
       */
      while (number != 0) {
        x[j++] = '0' + number % 10; // Converts the last digit to char and stores it.
        number /= 10;               // Removes the last digit from 'number'.
      }

      // Overwrites parts of the original string 's[i]' to form the abbreviated word.
      s[i][0] = first; // Ensures the first character is in place.
      k = 1;           // Start after the first character.
      /**
       * Block Logic: Appends the digits from 'x' (in correct order after reversal) to 's[i]'.
       * Invariant: The digits of the intermediate character count are placed in 's[i]' after the first character.
       */
      while (j > 0) {
        s[i][k++] = x[--j]; // Appends digits from 'x' in correct order.
      }
      s[i][k++] = last; // Appends the last character.

      /**
       * Block Logic: Null-terminates the abbreviated string by filling remaining characters with '\0'.
       * Invariant: The abbreviated word in 's[i]' is properly terminated.
       */
      for (int p = k; p < s[i].length(); p++)
        s[i][p] = '\0'; // Null-terminate any excess characters.

      j = 0; // Resets counters for next word.
      k = 0;
    }
  }

  /**
   * Block Logic: Prints all processed words (abbreviated or original) to standard output.
   * Invariant: Each word from 's' is printed on a new line.
   */
  for (int i = 0; i < n; i++) {
    cout << s[i] << endl;
  }

  return 0;
}
