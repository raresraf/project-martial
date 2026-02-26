/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem using C++ STL algorithms.
 * @author Deepesh Nimma
 * 
 * @details This program implements the "String Task" logic by first converting the
 * entire string to lowercase, then removing all vowels, and finally inserting a
 * '.' before each remaining character (consonant). This version demonstrates an
 * idiomatic C++ approach using the Standard Template Library.
 *
 * @algorithm
 * 1. Read the input into a `std::string`.
 * 2. Use `std::transform` with `::tolower` to convert the entire string to lowercase.
 * 3. For each vowel ('a', 'e', 'i', 'o', 'u', 'y'), use the erase-remove idiom
 *    (`s.erase(remove(s.begin(), s.end(), 'vowel'), s.end())`) to delete all
 *    of its occurrences from the string.
 * 4. Iterate through the resulting string of consonants and print each character,
 *    preceded by a '.'.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string. Each step
 * (`transform`, the series of `remove`/`erase` calls, and the final print loop)
 * takes time proportional to the length of the string.
 * Space Complexity: O(N) to store the string itself.
 */
// Written By Deepesh Nimma
// Codeforces: 118A Codeforces
#include <bits/stdc++.h>
using namespace std;

// Unused macros, likely part of a personal template.
#define ll long long
#define end1 "\n"

int main() {
  string s;
  cin >> s;

  // Use std::transform to apply the ::tolower function to every character in the
  // string, converting it to lowercase in-place.
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  /**
   * @block
   * @brief Remove all vowels from the string using the erase-remove idiom.
   * @details The `remove` algorithm shunts all non-vowel characters to the
   * beginning of the string and returns an iterator to the new logical end.
   * The `erase` member function then truncates the string at that iterator,
   * effectively deleting the vowels. This is performed for each vowel.
   */
  s.erase(remove(s.begin(), s.end(), 'a'), s.end());
  s.erase(remove(s.begin(), s.end(), 'e'), s.end());
  s.erase(remove(s.begin(), s.end(), 'i'), s.end());
  s.erase(remove(s.begin(), s.end(), 'o'), s.end());
  s.erase(remove(s.begin(), s.end(), 'u'), s.end());
  s.erase(remove(s.begin(), s.end(), 'y'), s.end());

  /**
   * @block
   * @brief Print the resulting string of consonants with preceding dots.
   */
  for (int i = 0; i < s.size(); ++i) {
    cout << '.' << s[i];
  }
  cout << endl; // Add a final newline for clean output.
}