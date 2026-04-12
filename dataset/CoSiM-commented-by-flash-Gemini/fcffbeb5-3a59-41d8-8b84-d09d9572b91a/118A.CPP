/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 *
 * This program transforms an input string according to specific rules:
 * 1. All vowels (a, o, y, e, u, i, case-insensitive) are removed.
 * 2. A '.' character is inserted before each consonant.
 * 3. All resulting characters are converted to lowercase.
 *
 * Algorithm: Character-by-character string processing.
 * Time Complexity: O(L) where L is the length of the input string, due to single pass iteration.
 * Space Complexity: O(L) for the input character array, with output printed directly.
 */
#include <iostream>
#include <string.h>
using namespace std;

/**
 * @brief Main function to perform the string transformation.
 *
 * Functional Utility: Reads the input string into a character array, iterates
 * through its characters, converts them to lowercase, filters out vowels,
 * and prints consonants prefixed with a dot.
 */
int main() {
  char arr[101]; // Declare a C-style character array to store the input string.
  int i, j;      // Loop counters.
  cin.getline(arr, 101); // Read the input line into the character array, up to 100 characters.
  int len = strlen(arr); // Get the actual length of the string.

  /**
   * Block Logic: Iterates through each character of the input array.
   * First, converts any uppercase letters to lowercase based on ASCII values.
   * Then, checks if the character is a consonant (not a, e, i, o, u, y).
   * If it's a consonant, prints a '.' followed by the character.
   * Invariant: All processed characters are either skipped (vowels) or transformed (consonants).
   */
  for (i = 0; i < len; i++) {
    // Convert uppercase character to lowercase using ASCII values.
    // 'A' (65) to 'Z' (90) converted to 'a' through 'z'.
    if (arr[i] >= 65 && arr[i] <= 90) {
      arr[i] = 'a' + (arr[i] - 'A');
    }
    // Check if the character is a consonant (i.e., not a, e, i, o, u, y).
    if (arr[i] != 'a' && arr[i] != 'e' && arr[i] != 'i' && arr[i] != 'o' &&
        arr[i] != 'u' && arr[i] != 'y') {
      cout << "." << arr[i]; // Print '.' followed by the consonant.
    }
  }
  cout << endl; // Print a newline character at the end.
}
