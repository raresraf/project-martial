/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" problem (Codeforces 112A).
 * @author competitive-programmer
 *
 * @details This program compares two strings lexicographically, ignoring differences
 * in case. It outputs -1 if the first string is lexicographically smaller than
 * the second, 1 if it is larger, and 0 if they are equal.
 *
 * Algorithm:
 * 1. Read the two input strings.
 * 2. Convert both strings entirely to a consistent case (e.g., lowercase).
 *    This ensures the comparison is case-insensitive.
 * 3. Use the standard lexicographical comparison operators on the modified strings.
 * 4. Print -1, 1, or 0 based on the comparison result.
 *
 * Time Complexity: O(L), where L is the length of the strings, for iterating
 * through them to convert case and then for the comparison.
 * Space Complexity: O(L) to store the two strings.
 */
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief An unused and flawed function intended to toggle character case.
 * @param c The character to toggle.
 * @return The toggled character.
 * @note This function is not called by main. Its logic is also incorrect as it
 * assumes a character is either upper or lower, but doesn't handle non-alphabetic
 * characters and would incorrectly "toggle" them.
 */
char toggleCase(char c) {
  int dif = 'a' - 'A';
  if (c <= 'Z')
    c += dif;
  else
    c -= dif;
  return c;
}

int main() {
  string s, a;
  // Read the two strings to be compared.
  cin >> s >> a;
  
  /**
   * Block Logic: Convert both strings to lowercase to ensure a case-insensitive
   * comparison.
   */
  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
    a[i] = tolower(a[i]);
  }
  
  /**
   * Decision Logic: Use the overloaded operators for std::string to perform
   * a lexicographical comparison on the standardized lowercase strings.
   */
  if (s < a)
    cout << -1;
  else if (s > a)
    cout << 1;
  else
    cout << 0;
    
  return 0;
}
