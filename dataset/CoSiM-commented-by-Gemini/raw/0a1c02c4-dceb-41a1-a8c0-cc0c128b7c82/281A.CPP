/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given word.
 * @details This program solves the "Word Capitalization" problem from Codeforces.
 * It reads a single word and capitalizes its first letter.
 *
 * @author Taewoo Kim
 * @date 3/23/2017
 *
 * @see https://codeforces.com/problemset/problem/281/A
 *
 * Algorithm:
 * 1. Read the input word into a std::string.
 * 2. Access the first character of the string via index 0.
 * 3. Use the `toupper()` function from `<cctype>` to convert the character to uppercase.
 * 4. The modification happens in-place.
 * 5. Print the modified string to standard output.
 *
 * Time Complexity: O(N), where N is the length of the word, dominated by I/O.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string> // Required for std::string
#include <cctype> // Required for toupper

using namespace std;

int main() {
  string s;
  cin >> s;

  // Access the first character and convert it to uppercase.
  s[0] = toupper(s[0]);

  cout << s;
  return 0;
}
