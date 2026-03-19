/**
 * @file 118A.CPP
 * @brief Modern C++ solution for the "String Task" problem (Codeforces 118A).
 * @details This program processes an input string by applying three rules:
 * 1. It removes all vowels ('A', 'O', 'Y', 'E', 'U', 'I', and their lowercase counterparts).
 * 2. It inserts a '.' character before every consonant.
 * 3. It converts all consonants to their lowercase form.
 * This implementation uses modern C++ features like std::string and range-based for loops.
 *
 * Algorithm: String Building
 * The program iterates through each character of the input string. It uses a helper
 * function, `isvowel`, to determine if a character is a vowel. If it is, the character
 * is ignored. If it's a consonant, a '.' is appended to a result string, followed by the
 * lowercase version of the consonant. After checking all characters, the final result
 * string is printed.
 *
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass.
 * Space Complexity: O(N), to store both the input and the resulting `ans` string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Checks if a character is a vowel.
 * @param a The character to check.
 * @return True if the character is a vowel (case-insensitive, including 'y'), false otherwise.
 */
bool isvowel(char a) {
  if (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'A' ||
      a == 'E' || a == 'I' || a == 'O' || a == 'U' || a == 'y' || a == 'Y')
    return true;
  else
    return false;
}

int main() {
  // t: the input string.
  // ans: the string built according to the problem's rules.
  string t, ans = "";
  cin >> t;
  
  /**
   * Block Logic: Iterate through each character of the input string.
   * This uses a range-based for loop, a modern C++ feature for cleaner iteration.
   */
  for (auto it : t) {
    // If the character is a vowel, skip it and move to the next one.
    if (isvowel(it))
      continue;
    else {
      // If the character is a consonant, append a '.' to the result.
      ans += '.';
      // Then, append the lowercase version of the consonant.
      ans += tolower(it);
    }
  }
  
  // Print the final transformed string.
  cout << ans;
}
