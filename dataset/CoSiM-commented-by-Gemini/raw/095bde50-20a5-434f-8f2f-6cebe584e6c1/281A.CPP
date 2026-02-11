/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given word.
 * @details This program solves the "Word Capitalization" problem from Codeforces.
 * It reads a single word and ensures its first character is uppercase, leaving
 * the rest of the word unchanged.
 *
 * @see https://codeforces.com/problemset/problem/281/A
 *
 * Algorithm:
 * 1. Read the entire word into a std::string.
 * 2. Access the first character of the string using index 0.
 * 3. Convert this character to its uppercase equivalent using `toupper()`.
 * 4. Assign the uppercase character back to the first position in the string.
 * 5. Print the modified string.
 *
 * Time Complexity: O(N), where N is the length of the word, dominated by the
 * input and output operations.
 * Space Complexity: O(N) to store the word in a string.
 */
#include <iostream>
#include <string> // For std::string
#include <cctype> // For toupper

using namespace std;

int main() {
  string line;
  
  // Read the input word.
  getline(cin, line);
  
  // Capitalize the first letter of the word.
  // The toupper function converts a character to uppercase.
  line[0] = toupper(line[0]);
  
  // Print the modified string.
  cout << line << endl;
  
  return 0;
}
