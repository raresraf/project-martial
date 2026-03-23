/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A, "Word Capitalization".
 * @details The program reads a single word and capitalizes its first letter.
 * If the first letter is already capitalized, it remains unchanged. The rest
 * of the word is not modified.
 *
 * @algorithm String manipulation
 * @complexity
 * Time: O(1), as it only modifies the first character of the string. The time
 * to read and write the string is O(L) where L is the length, but the core
 * logic is constant time.
 * Space: O(L) to store the string.
 */
#include <iostream>
#include <string>   // For std::string
#include <cctype>   // For std::toupper

using namespace std;

int main() {
  string s;
  
  // Read the input word.
  cin >> s;

  // Check if the string is not empty to prevent errors.
  if (!s.empty()) {
    // Access the first character and convert it to uppercase.
    s[0] = toupper(s[0]);
  }

  // Print the modified string.
  cout << s << endl;

  return 0; // Indicate successful execution.
}