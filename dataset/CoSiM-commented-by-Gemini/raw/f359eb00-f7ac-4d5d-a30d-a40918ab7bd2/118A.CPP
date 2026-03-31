/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem by transforming a string.
 *
 * This program performs a series of transformations on an input string as per
 * the "String Task" problem rules:
 * 1. All uppercase letters are converted to lowercase.
 * 2. All vowels ('a', 'o', 'y', 'e', 'u', 'i') are deleted.
 * 3. A '.' character is inserted before each consonant.
 * The final transformed string is then printed.
 *
 * @author f359eb00-f7ac-4d5d-a30d-a40918ab7bd2
 *
 * @algorithm String Traversal and Transformation
 * @complexity
 * Time: O(N), where N is the length of the input string, as it iterates
 *       through the string once.
 * Space: O(N), to store the resulting transformed string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {

  string original, ret;

  cin >> original;

  int i;
  /**
   * @brief Main loop to iterate through each character of the original string.
   */
  for (i = 0; i < original.length(); i++) {
    /**
     * @brief Convert uppercase letters to lowercase.
     * It checks if the character's ASCII value is in the uppercase range (65-90)
     * and adds 32 to convert it to its lowercase equivalent.
     */
    if (original[i] >= 65 && original[i] <= 90) {
      original[i] += 32;
    }
    
    /**
     * @brief Check if the character is a consonant.
     * The character is considered a consonant if it is not one of the specified vowels.
     */
    if (original[i] != 'a' && original[i] != 'o' && original[i] != 'y' &&
        original[i] != 'e' && original[i] != 'u' && original[i] != 'i') {
      // If it's a consonant, prepend a '.' and then the character to the result string.
      ret.push_back('.');
      ret.push_back(original[i]);
    }
  }

  // Print the final transformed string.
  cout << ret << "
";

  return 0;
}