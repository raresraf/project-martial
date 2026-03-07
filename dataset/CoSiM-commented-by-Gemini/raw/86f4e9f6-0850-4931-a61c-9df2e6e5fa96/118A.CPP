/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem using a C++ `for_each` loop.
 *
 * @details
 * This program modifies a string by deleting all vowels, inserting a '.' before
 * each consonant, and converting all consonants to lowercase. This implementation
 * uses a modern C++ approach with a `for_each` loop and a lambda function.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Use `std::for_each` to iterate over each character of the string.
 * 3. In the loop, for each character:
 *    a. Convert the character to lowercase.
 *    b. Check if the character is a consonant. The helper function `isConsonant`
 *       does this by checking if the character is *not* in a predefined list of vowels.
 *    c. If it is a consonant, print a '.' followed by the character.
 *
 * Time Complexity: O(N), where N is the length of the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

// NOTE: This global array is misnamed. It contains VOWELS, not consonants.
char consonants[6] = {'a', 'e', 'i', 'o', 'u', 'y'};

/**
 * @brief Checks if a character is a consonant by testing if it's not a vowel.
 * @param a The lowercase character to check.
 * @return True if 'a' is a consonant, false if it is a vowel.
 */
bool isConsonant(char a) {
  // Iterates through the global array of vowels.
  for (char c : consonants) {
    if (a == c)
      return false; // It's a vowel.
  }
  return true; // It's a consonant.
}

int main() {

  string input;

  cin >> input;

  /**
   * @brief Iterates over each character of the string using std::for_each and a lambda.
   * The lambda processes each character to produce the desired output.
   */
  for_each(input.begin(), input.end(), [](char &c) {
    // Convert character to lowercase. The reference `&c` modifies the string in-place,
    // though the modified string itself is not used later.
    c = ::tolower(c);
    if (isConsonant(c)) {
      cout << '.' << c;
    }
  });

  cout << endl;

  return 0;
}
