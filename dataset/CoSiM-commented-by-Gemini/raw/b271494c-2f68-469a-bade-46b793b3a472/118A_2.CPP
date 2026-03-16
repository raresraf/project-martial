/**
 * @file 118A_2.CPP
 * @brief Solution to Codeforces problem 118A - "String Task".
 * @details This program processes a string according to specific rules:
 *          1. It converts all uppercase letters to lowercase.
 *          2. It removes all vowels ('a', 'e', 'i', 'o', 'u', 'y').
 *          3. It inserts a "." character before each consonant.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. Convert the character to lowercase.
 * 4. Check if the character is a vowel.
 * 5. If it is not a vowel (i.e., it is a consonant), print a '.' followed by the character.
 *
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass over the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {
  int n;
  char array[105];
  cin >> array;

  /**
   * @brief Main loop to iterate over each character of the input string.
   * The loop continues for the entire length of the string.
   */
  for (int i = 0; i < strlen(array); i++) {
    /**
     * @brief Pre-condition: Ensure character is in lowercase for uniform processing.
     * This step standardizes the input to handle both 'A' and 'a' as the same vowel.
     */
    if (isupper(array[i])) {
      array[i] = tolower(array[i]);
    }

    /**
     * @brief Block Logic: Filter out vowels.
     * The vowels are defined as 'a', 'e', 'o', 'i', 'u', and 'y' for this problem.
     * If the character is not a vowel, it is processed as a consonant.
     */
    if (array[i] != 'a' && array[i] != 'e' && array[i] != 'o' &&
        array[i] != 'i' && array[i] != 'u' && array[i] != 'y') {
      // For each consonant, prepend a '.' and print the character.
      cout << "." << array[i];
    }
  }

  return 0;
}
