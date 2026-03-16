/**
 * @file 118A.CPP
 * @brief An efficient, single-pass solution to the "String Task" problem (Codeforces 118A).
 *
 * @details
 * This program reformats a string based on the following rules:
 * 1. Deletes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 2. Inserts a '.' before each consonant.
 * 3. Converts all consonants to lowercase.
 *
 * This implementation is a clean and efficient single-pass solution that uses modern
 * C++ features. It iterates through the string once, performing all transformations
 * in a single step for each character.
 *
 * An `std::unordered_set` is used to store the vowels, allowing for very fast,
 * average O(1) time complexity for checking if a character is a vowel.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Create a hash set of lowercase vowel characters.
 * 3. For each character in the input string:
 *    a. Convert the character to its lowercase equivalent.
 *    b. Check if the character exists in the vowel set.
 *    c. If it does not, print a '.' followed by the lowercase character.
 *
 * Time Complexity: O(L), where L is the length of the string, due to the single pass.
 * Space Complexity: O(L) to store the input string. The vowel set uses O(1) space.
 */
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <cctype> // For tolower

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string input;
  cin >> input;
  
  // Use a hash set for efficient, O(1) average time complexity vowel lookups.
  std::unordered_set<char> vowels = {'a', 'i', 'u', 'e', 'o', 'y'};

  /**
   * @brief Processes the string in a single pass.
   * @details This loop iterates through each character of the input string. In each
   * iteration, it converts the character to lowercase, checks if it's a consonant
   * using the hash set, and if so, prints the formatted output immediately.
   */
  for (auto it = input.begin(); it != input.end(); ++it) {
    // First, convert the character to lowercase to simplify the vowel check.
    (*it) = tolower(*it);
    
    // Check if the lowercase character is in the vowel set.
    // vowels.find() is an O(1) average time operation.
    if (vowels.find(*it) == vowels.end()) {
      // If not found, it's a consonant. Print the required output.
      cout << "." << *it;
    }
  }
  return 0;
}