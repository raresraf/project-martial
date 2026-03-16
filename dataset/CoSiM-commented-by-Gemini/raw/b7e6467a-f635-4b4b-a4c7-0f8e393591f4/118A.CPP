/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem (Codeforces 118A).
 *
 * @details
 * This program processes a given string based on a specific set of rules:
 * 1. All vowels ('A', 'O', 'Y', 'E', 'U', 'I', and their lowercase counterparts)
 *    are deleted.
 * 2. Before each consonant, a '.' character is inserted.
 * 3. All uppercase consonants are replaced with their corresponding lowercase versions.
 *
 * The code iterates through each character of the input string. For each character,
 * it checks if it is a vowel by comparing it against a predefined list of vowel
 * characters. If the character is not a vowel (i.e., it's a consonant), the program
 * prints a '.' followed by the lowercase version of that consonant.
 *
 * The implementation uses C-style strings and I/O (`char[]`, `fscanf`, `fprintf`).
 *
 * Algorithm:
 * 1. Read an input string.
 * 2. For each character in the string:
 *    a. Check if the character is a vowel (case-insensitive).
 *    b. If it is not a vowel, print a '.' and the lowercase version of the character.
 *
 * Time Complexity: O(L), where L is the length of the string. Although there is a
 *                  nested loop, the inner loop runs a small, constant number of times.
 * Space Complexity: O(L), to store the input string.
 */
#include <cctype>
#include <cstdio>
#include <cstring> // Required for strlen

using namespace std;

int main() {
  char s[101]; // Increased size to be safe with 100 char limit + null terminator
  char a[] = "AOYEUIaoyeui"; // Removed size 13 to let compiler deduce it
  int found = 0;
  char c;

  // Read a single word from standard input.
  fscanf(stdin, "%100s", s);

  /**
   * @brief Main loop to process each character of the input string.
   */
  for (int i = 0; s[i] != '\0'; ++i) {
    /**
     * @brief Vowel Check: Determines if the current character is a vowel.
     * @details This inner loop compares the current character `s[i]` against
     * a predefined list of vowels. The `found` flag is set if a match occurs.
     */
    for (int j = 0; a[j] != '\0'; ++j) {
      if (s[i] == a[j]) {
        found = 1;
        break;
      }
    }

    // If `found` is 0, the character is a consonant.
    if (!found) {
      c = s[i];
      // Print a '.' followed by the lowercase version of the consonant.
      fprintf(stdout, ".%c", tolower(c));
    }
    
    // Reset the flag for the next character.
    found = 0;
  }
}