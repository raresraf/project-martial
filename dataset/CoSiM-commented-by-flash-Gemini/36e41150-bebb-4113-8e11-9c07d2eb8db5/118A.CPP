/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem: "String Task".
 *
 * This program transforms an input string according to specific rules:
 * 1. All vowels (A, O, Y, E, U, I, case-insensitive) are removed.
 * 2. All consonants are converted to lowercase.
 * 3. A period (".") is inserted before each consonant.
 *
 * Algorithm: Character-by-character string processing with conditional logic.
 * Time Complexity: O(L), where L is the length of the input string, as it iterates through the string once.
 */
#include <iostream> // Functional Utility: Included for standard input/output operations.
#include <string>   // Functional Utility: Included for string manipulation.
#include <cctype>   // Functional Utility: Included for character manipulation functions like tolower.

using namespace std; // Functional Utility: Uses the standard namespace to avoid prefixing standard library elements with std::

string a; // Global Variable: Stores the input string.

int main() {
  // Functional Utility: Optimizes standard I/O streams for faster execution,
  // crucial in competitive programming for time-sensitive tasks.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Input: Reads the input string into global variable 'a'.
  cin >> a;

  // Block Logic: Iterates through each character of the input string.
  // Invariant: For each iteration, `a[i]` is the current character being processed.
  for (int i = 0; i < a.length(); i++) {
    // Block Logic: Checks if the current character is a vowel (case-insensitive).
    // Precondition: `a[i]` is a character from the input string.
    if (a[i] == 'A' || a[i] == 'O' || a[i] == 'Y' || a[i] == 'E' ||
        a[i] == 'U' || a[i] == 'I' || a[i] == 'a' || a[i] == 'o' ||
        a[i] == 'y' || a[i] == 'e' || a[i] == 'u' || a[i] == 'i') {
      // Output: Vowels are effectively "removed" by printing an empty string.
      cout << "";
    } else if (a[i] == 'B' || a[i] == 'C' || a[i] == 'D' || a[i] == 'F' ||
               a[i] == 'G' || a[i] == 'H' || a[i] == 'J' || a[i] == 'K' ||
               a[i] == 'L' || a[i] == 'M' || a[i] == 'N' || a[i] == 'P' ||
               a[i] == 'Q' || a[i] == 'R' || a[i] == 'S' || a[i] == 'T' ||
               a[i] == 'W' || a[i] == 'V' || a[i] == 'X' || a[i] == 'Z') {
      // Block Logic: Converts an uppercase consonant to lowercase and prints it prefixed with a dot.
      // Precondition: The character is an uppercase consonant.
      a[i] = (char)tolower(a[i]); // Inline: Converts the uppercase consonant to its lowercase equivalent.
      cout << "." << a[i];
    } else if (a[i] == 'b' || a[i] == 'c' || a[i] == 'd' || a[i] == 'f' ||
               a[i] == 'g' || a[i] == 'h' || a[i] == 'j' || a[i] == 'k' ||
               a[i] == 'l' || a[i] == 'm' || a[i] == 'n' || a[i] == 'p' ||
               a[i] == 'q' || a[i] == 'r' || a[i] == 's' || a[i] == 't' ||
               a[i] == 'w' || a[i] == 'v' || a[i] == 'x' || a[i] == 'z') {
      // Block Logic: Prints a lowercase consonant prefixed with a dot.
      // Precondition: The character is a lowercase consonant.
      cout << "." << a[i];
    }
  }
  return 0; // Functional Utility: Indicates successful program execution.
}
