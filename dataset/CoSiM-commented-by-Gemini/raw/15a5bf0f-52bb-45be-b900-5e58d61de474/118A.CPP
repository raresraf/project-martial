/**
 * @file 118A.CPP
 * @brief An inefficient and redundant solution for the "String Task" problem.
 * @details The program attempts to solve the "String Task" problem by first deleting
 * all vowels, then converting consonants to lowercase and prepending a '.' to them.
 * The implementation is notable for its inefficiencies and redundant logic.
 *
 * The intended rules are:
 * 1. Convert all letters to lowercase.
 * 2. Delete all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 3. Insert a '.' before each consonant.
 *
 * @section algorithm Algorithm as Implemented
 * 1. An array of both upper and lowercase vowels is defined.
 * 2. The input string is read.
 * 3. **First Pass (Vowel Deletion):** The code iterates through the string. For each
 *    character, it checks against the vowel array. If a vowel is found, it's erased
 *    from the string in-place. The loop index is decremented to account for the string's shift.
 * 4. **Second Pass (Redundant Deletion and Processing):** The code enters a second, similar
 *    loop over the already-modified string.
 *    a. It *redundantly* repeats the vowel deletion logic.
 *    b. For each character (which should now only be consonants), it checks if it's uppercase
 *       using its ASCII value.
 *    c. It converts the character to lowercase if needed.
 *    d. It prints a '.' followed by the lowercase consonant.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N^2), where N is the length of the string. The `string::erase` operation inside a loop is costly.
 * - Space Complexity: O(1), as all modifications are done in-place on the input string (besides the initial storage).
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // Array containing both upper and lower case vowels for matching.
  char vowels[12] = {'a', 'o', 'y', 'e', 'u', 'i',
                     'A', 'O', 'Y', 'E', 'U', 'I'};
  string input;
  cin >> input;
  /**
   * @brief First pass: Remove all vowels from the input string.
   * This loop iterates through the string, and upon finding a vowel,
   * erases it. The index `i` is decremented to re-examine the
   * character that shifts into the current position after erasure.
   */
  for (int i = 0; i < input.length(); i++) {
    for (int j = 0; j < 12; j++) {
      char temp = input.at(i);
      char vowel = vowels[j];
      if (temp == vowel) {
        input.erase(i, 1);
        i--;
        break;
      }
    }
  }
  /**
   * @brief Second pass: Redundantly checks for vowels again, converts to lowercase,
   * and prepends dots before printing.
   */
  for (int i = 0; i < input.length(); i++) {
    // This inner loop is redundant as the first pass should have removed all vowels.
    for (int j = 0; j < 12; j++) {
      char temp = input.at(i);
      char vowel = vowels[j];
      if (temp == vowel) {
        input.erase(i, 1);
        break;
      }
    }
    char temp = input.at(i);
    // Process the character (which should be a consonant at this point).
    // It is converted to lowercase and printed with a prepended dot.
    if (temp >= 65 && temp < 97) { // Check for uppercase using ASCII values.
      temp = temp + 32;
      cout << "." << temp;
    } else {
      cout << "." << temp;
    }
  }
  return 0;
}