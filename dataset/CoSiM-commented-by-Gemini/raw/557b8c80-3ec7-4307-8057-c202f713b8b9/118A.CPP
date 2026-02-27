/*
 * Problem description
 *
 * https://codeforces.com/problemset/problem/118/A
 *
 */

/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details The task requires processing a string by deleting all vowels,
 * inserting a '.' before each consonant, and converting all consonants to
 * lowercase.
 *
 * @algorithm
 * This solution uses modern C++ algorithms and a stringstream to build the result.
 * 1. Read the input string.
 * 2. Convert the entire string to lowercase using `std::for_each` with a lambda.
 * 3. Define an array of vowel characters.
 * 4. Iterate through the lowercase input string. For each character:
 *    a. Use `std::find` to check if the character exists in the vowel array.
 *    b. If it's not a vowel (it's a consonant), append a '.' and the
 *       consonant to a `stringstream`.
 * 5. After the loop, extract the content from the `stringstream` back into the
 *    original string variable, overwriting it.
 * 6. Print the modified string.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string. Both the
 * `for_each` and the main loop are linear passes. `std::find` on the small,
 * fixed-size vowel array is effectively a constant-time operation.
 * Space Complexity: O(L) for the string, stringstream, and vowel array.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  stringstream ss; // A stringstream to build the output string.
  cin >> a;
  
  char ac; // To hold the current character.
  char v[6] = {'a', 'o', 'y', 'e', 'u', 'i'}; // Array of vowels.

  // Use a lambda with for_each to convert the entire string to lowercase.
  for_each(begin(a), end(a), [](char &c) { c = tolower(c); });
  
  /**
   * @block
   * @brief Process each character, adding consonants prefixed with '.' to the stringstream.
   */
  for (int i = 0; a[i] != '\0'; i++) {
    ac = a[i];
    // Check if the character is present in the vowel array 'v'.
    bool ia = find(begin(v), end(v), ac) != end(v);
    
    // If it's not a vowel, process it.
    if (!ia)
      ss << '.' << ac;
  }
  
  // Extract the built string from the stringstream back into 'a'.
  // Note: A more common and direct way to get the result is `ss.str()`.
  ss >> a;
  cout << a << endl;
  ; // This is a stray, harmless semicolon.
  return 0;
}
