/**
 * @6c508edb-10a5-4ee5-b256-e5f08a7e8b71/71A.CPP
 * @brief Solution for a competitive programming problem involving word abbreviation.
 *
 * This program implements a word abbreviation logic. For each input word, if its
 * length exceeds 10 characters, it is abbreviated to its first character, followed
 * by the count of characters between the first and last, and then its last character.
 * Words with length 10 or less are printed as is.
 * Example: "localization" becomes "l10n".
 *
 * Algorithm:
 * 1. Read the integer 'n', representing the number of words to process.
 * 2. Iterate 'n' times:
 *    a. Dynamically allocate memory for a character array (C-style string) to hold the word.
 *    b. Read the current word into the allocated memory.
 *    c. Check the length of the word using strlen().
 *    d. If the length is greater than 10:
 *       i. Print the first character.
 *       ii. Print the length of the word minus 2 (representing the abbreviated middle part).
 *       iii. Print the last character.
 *    e. Else (length is 10 or less):
 *       i. Print the word as it is.
 *    f. Free the dynamically allocated memory for the word.
 *
 * Time Complexity: O(L_total) where L_total is the sum of lengths of all input words,
 *                  due to reading words and strlen() operations.
 * Space Complexity: O(L_max) where L_max is the maximum length of a single word,
 *                   for storing one word at a time.
 */

#include <iostream>  // Functional Utility: Provides std::cin and std::cout for input/output.
#include <stdio.h>   // Functional Utility: Provides standard input/output functions, though not directly used in this C++ code.
#include <stdlib.h>  // Functional Utility: Provides memory allocation functions like malloc and free.
#include <string.h>  // Functional Utility: Provides C-style string manipulation functions like strlen.
#include <tgmath.h>  // Functional Utility: Provides type-generic mathematical functions, not used in this specific logic.

using namespace std;

int main(void) {
  int n;
  // Functional Utility: Reads the number of words to be processed.
  cin >> n;

  // Block Logic: Iterates 'n' times to process each word.
  // Pre-condition: 'n' is a positive integer.
  // Invariant: Each word will be read and either abbreviated or printed as is.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Dynamically allocates memory for a character array to store the word.
    // Inline: Allocates space for up to 100 characters plus a null terminator.
    char *word = (char *)malloc(101 * sizeof(char)); // Fixed sizeof(word) to sizeof(char)

    // Functional Utility: Reads the current word into the allocated memory.
    cin >> word;

    // Conditional Logic: Checks if the word's length exceeds 10 characters for abbreviation.
    if (strlen(word) > 10) {
      // Functional Utility: Prints the first character, abbreviated length, and last character.
      cout << word[0] << strlen(word) - 2 << word[strlen(word) - 1] << endl;
    } else {
      // Functional Utility: Prints the word as is if it's not too long.
      cout << word << endl;
    }

    // Functional Utility: Frees the dynamically allocated memory to prevent memory leaks.
    free(word);
  }

  return 0; // Functional Utility: Indicates successful program execution.
}
