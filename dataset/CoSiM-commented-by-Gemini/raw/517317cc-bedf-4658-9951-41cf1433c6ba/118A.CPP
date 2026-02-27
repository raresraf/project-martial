/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces using C-style I/O.
 * @details The problem requires processing a string with these rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Insert a '.' before each consonant.
 * 3. Convert all uppercase consonants to lowercase.
 * This implementation processes the string character by character and prints
 * the output directly without building an intermediate result string.
 *
 * @algorithm
 * 1. Read the input string into a character array.
 * 2. Iterate through the array until the null terminator is found.
 * 3. In each iteration:
 *    a. Convert the current character to lowercase.
 *    b. Check if the character is a consonant by ensuring it's not a vowel
 *       ('a', 'e', 'i', 'o', 'u', 'y').
 *    c. If it is a consonant, print a '.' followed by the character.
 * 4. After the loop, the full output will have been printed to standard out.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to the single
 * pass through the characters.
 * Space Complexity: O(L_max) for the character array buffer, where L_max is
 * the maximum possible length (statically set to 101 here).
 */
#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  // A fixed-size C-style string to hold the input.
  // Note: Using scanf with %s on a fixed buffer is unsafe and can lead
  // to buffer overflows if the input is too long.
  char x[101];
  scanf("%s", x); // The '&' is not necessary for char arrays with scanf.

  /**
   * @block
   * @brief Iterate through the input string character by character.
   * The loop continues until the null terminator (0) is reached.
   */
  for (int i = 0; x[i] != 0; i++) {
    // Convert the character to its lowercase equivalent.
    x[i] = tolower(x[i]);
    
    /**
     * @brief Check if the character is a consonant.
     * Vowels are 'a', 'e', 'i', 'o', 'u', and 'y' for this problem.
     */
    if (x[i] != 'a' && x[i] != 'e' && x[i] != 'i' && x[i] != 'o' &&
        x[i] != 'u' && x[i] != 'y') {
      // If it's a consonant, print a '.' followed by the character.
      printf(".%c", x[i]);
    }
  }
  return 0;
}
