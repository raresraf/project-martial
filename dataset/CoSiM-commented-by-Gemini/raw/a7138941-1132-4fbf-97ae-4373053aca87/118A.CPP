/**
 * @file 118A.CPP
 * @brief Solution for CodeForces Problem 118A - "String Task".
 * 
 * @details
 * This program processes a string according to a specific set of rules:
 * 1. It deletes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 2. It inserts a character "." before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding lowercase ones.
 *
 * @algorithm String processing with vowel lookup
 * 1. Create a vector containing all lowercase vowels ('a', 'e', 'i', 'o', 'u', 'y').
 * 2. Iterate through each character of the input string.
 * 3. For each character, convert it to its lowercase equivalent.
 *    - This is done using ASCII arithmetic (`s[i] + 32`), which is less portable than `tolower()`.
 * 4. Use `std::find` to check if the lowercase character exists in the vowel vector.
 * 5. If the character is not found in the vowel vector (i.e., it's a consonant):
 *    a. Append a '.' to the result string.
 *    b. Append the lowercase consonant to the result string.
 * 6. Print the final result string.
 *
 * @time_complexity O(L), where L is the length of the input string. Each `std::find` on the
 *                  small vowel vector takes constant time on average.
 * @space_complexity O(L) for storing the input and output strings.
 */
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cctype> // For tolower()

using namespace std;

int main() {
  string s;
  cin >> s;
  string ans = ""; // The result string.
  
  // A vector to store the vowels for easy lookup.
  vector<char> v;
  v.push_back('a');
  v.push_back('e');
  v.push_back('i');
  v.push_back('o');
  v.push_back('u');
  v.push_back('y');

  /**
   * @brief Main loop to iterate through the input string and apply transformation rules.
   */
  for (int i = 0; i < s.length(); i++) {
    // Convert the current character to lowercase.
    char temp = tolower(s[i]);
    
    /**
     * @brief Check if the character is a consonant by searching for it in the vowel vector.
     * `find` returns `v.end()` if the element is not found.
     */
    if (find(v.begin(), v.end(), temp) == v.end()) {
      // If it's a consonant, prepend a '.' and append the lowercase consonant.
      ans += ".";
      ans += temp;
    }
  }
  
  cout << ans;
  return 0;
}
