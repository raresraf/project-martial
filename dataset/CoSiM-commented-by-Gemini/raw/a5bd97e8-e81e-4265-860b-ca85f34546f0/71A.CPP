/**
 * @file 71A.CPP
 * @brief Solution for CodeForces Problem 71A - "Way Too Long Words".
 * 
 * @details
 * This program processes a list of words. For each word, if its length is greater than 10,
 * it is abbreviated. The abbreviation consists of the first letter, the number of letters
 * between the first and last, and the last letter. Words with 10 or fewer letters are
 * printed unchanged.
 *
 * @algorithm String processing and abbreviation
 * @time_complexity O(N * L), where N is the number of words and L is the maximum length of a word.
 *                  The complexity is dominated by reading the words and calculating their length.
 * @space_complexity O(L) for storing the longest word.
 *
 * @note The use of a fixed-size `char` array `ch[100]` is a potential buffer overflow risk
 * if an input word exceeds 99 characters. A `std::string` would be safer. The inner loop
 * for counting characters is equivalent to `strlen(ch)`.
 */
#include <iostream>
#include <string.h> // For strlen

using namespace std;

int main() {
  char ch[101]; // Character array to store the word, size 101 for up to 100 chars + null terminator.
  int i, j, line, count;
  cin >> line; // Read the number of words to process.

  /**
   * @brief Main loop to process each word.
   */
  for (i = 1; i <= line; i++) {
    cin >> ch;
    
    // Calculate the length of the string.
    count = strlen(ch);
    
    /**
     * @brief Logic to decide whether to abbreviate the word.
     * Pre-condition: `count` holds the length of the word in `ch`.
     */
    if (count > 10) {
      // Abbreviate: first letter, count of middle letters, last letter.
      cout << ch[0] << count - 2 << ch[count - 1] << endl;
    } else {
      // Print the word as is.
      cout << ch << endl;
    }
  }
  return 0;
}
