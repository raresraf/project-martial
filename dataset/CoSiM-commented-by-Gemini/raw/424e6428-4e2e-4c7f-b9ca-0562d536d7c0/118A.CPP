/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *
 * @algorithm
 * The problem requires processing a string with the following rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Before each consonant, insert a character ".".
 * 3. All uppercase consonants are replaced with their corresponding lowercase ones.
 *
 * This solution builds a new result string by iterating through the input:
 * 1. A `std::set` of vowels is created for efficient O(log N) lookup.
 * 2. The code iterates through each character of the input string.
 * 3. Each character is converted to lowercase.
 * 4. It's checked if the character is a vowel by looking it up in the set.
 * 5. If the character is not found in the vowel set (i.e., it's a consonant),
 *    a '.' and the lowercase consonant are appended to a `result` string.
 * 6. After the loop, the fully constructed `result` string is printed.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the input string. Each
 *   character is processed once, and set lookup is very fast (logarithmic in
 *   the small, constant size of the vowel set).
 * - Space Complexity: O(L) to store both the input and the result strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string task;
  cin >> task;
  // The line below is commented out and not valid C++ for string case conversion.
  // task = tolower(task);
  string result = "";
  set<char> vowels = {'a', 'e', 'i', 'o', 'u', 'y'};
  /**
   * @brief Iterates through each character of the input string.
   */
  for (auto s : task) {
    s = tolower(s); // Convert character to lowercase.
    /**
     * @brief Checks if the character is a consonant.
     * The `vowels.find(s) == vowels.end()` is the key check. If a character
     * is not in the vowel set, it's treated as a consonant.
     */
    if (s >= 'b' && s <= 'z' && vowels.find(s) == vowels.end()) {
      // Append a dot and the lowercase consonant to the result.
      result += '.';
      result += s;
    }
  }
  // Print the final transformed string.
  cout << result << "\n";
}