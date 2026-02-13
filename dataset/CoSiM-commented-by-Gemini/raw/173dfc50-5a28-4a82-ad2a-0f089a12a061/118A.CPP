/**
 * @file 118A.CPP
 * @brief A character-by-character, stream-based solution for the "String Task" problem.
 * @details This program transforms an input string by removing vowels, converting consonants
 * to lowercase, and prepending a '.' to each consonant. This implementation is highly
 * memory-efficient as it processes the input one character at a time from the stream,
 * printing the result directly without building an intermediate string.
 *
 * @section algorithm Algorithm
 * 1. The program enters a loop that reads one character at a time from standard input
 *    until the stream ends (`while (cin >> x)`). This method automatically skips
 *    any whitespace between characters.
 * 2. For each character `x` read from the stream:
 *    a. It is converted to its lowercase equivalent if it is an uppercase letter.
 *    b. It is then checked against the set of vowels ('a', 'e', 'i', 'o', 'u', 'y').
 *    c. If it is a vowel, the empty statement `;` is executed, and nothing is printed.
 *    d. If it is not a vowel (i.e., a consonant), a '.' character is printed to standard
 *       output, followed by the lowercase consonant.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the total number of non-whitespace characters in the input stream.
 * - Space Complexity: O(1), as it only stores a single character at a time.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  char x;
  // This loop reads the input character by character, skipping whitespace,
  // until the end of the input stream is reached.
  while (cin >> x) {
    // Convert character to lowercase if it's an uppercase letter.
    if (x >= 'A' && x <= 'Z')
      x = x + 32;
    // Check if the now-lowercase character is a vowel.
    if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'y')
      ; // If it is a vowel, do nothing.
    else
      // If it is a consonant, print a '.' followed by the character.
      cout << '.' << x;
  }
}