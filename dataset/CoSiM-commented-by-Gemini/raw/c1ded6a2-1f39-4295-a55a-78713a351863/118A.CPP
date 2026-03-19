/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem.
 * @details This program processes a string according to a set of rules:
 * 1. It removes all vowels ('A', 'O', 'Y', 'E', 'U', 'I', and their lowercase counterparts).
 * 2. It inserts a '.' character before every consonant.
 * 3. It converts all uppercase consonants to their lowercase equivalents.
 *
 * Algorithm: Character-by-Character Processing
 * The program iterates through the input string. For each character, it checks if it is a vowel
 * by comparing it against a predefined list of vowel characters. If the character is not a vowel
 * (i.e., it is a consonant), the program prepends a '.' to it and converts it to lowercase
 * before printing it to the standard output.
 *
 * Time Complexity: O(N), where N is the length of the input string. The inner loop runs a fixed number of times (12), so it's a constant factor.
 * Space Complexity: O(N) for storing the input string `s`.
 */

#include <bits/stdc++.h>

using namespace std;

#define ll long long

// An array containing all vowel characters, both uppercase and lowercase, for easy checking.
char v[12] = {'a', 'o', 'y', 'e', 'u', 'i', 'A', 'O', 'Y', 'E', 'U', 'I'};

/**
 * @brief Converts an uppercase character to its lowercase equivalent.
 * @param t The character to convert.
 * @return The lowercase version of the character if it was uppercase, otherwise the character itself.
 */
char down(char t) {
  // ASCII 'a' is 32 greater than 'A'.
  if (t <= 'Z')
    return t + 32;
  else
    return t;
}

int main() {
  string s;
  cin >> s;
  bool flag;

  /**
   * Block Logic: Iterate over each character of the input string to process it.
   * Pre-condition: The string 's' contains the user input.
   */
  for (int i = 0; i < s.length(); i++) {
    // 'flag' is used to identify if a character is a consonant. Initialize to true.
    flag = true;
    
    /**
     * Block Logic: Check if the current character is a vowel.
     * Pre-condition: s[i] is the character to be checked.
     * Invariant: The loop checks s[i] against each vowel in the 'v' array.
     */
    for (int j = 0; j < 12; j++) {
      if (s[i] == v[j]) {
        // If a match is found, it's a vowel. Set flag to false and exit the inner loop.
        flag = false;
        break;
      }
    }

    // If 'flag' is still true after the inner loop, the character is a consonant.
    if (flag) {
      // Print a '.' followed by the lowercase version of the consonant.
      cout << "." << down(s[i]);
    }
  }
  return 0;
}