/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * @details This program processes a string according to specific rules:
 *          1. It deletes all the vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 *          2. It inserts a character "." before each consonant.
 *          3. It replaces all uppercase consonants with corresponding lowercase ones.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. For each character, check if it is a vowel (case-insensitive, including 'Y').
 * 4. If it is a vowel, skip it.
 * 5. If it is a consonant, append a '.' to the result string, followed by the
 *    lowercase version of the consonant.
 * 6. Print the final resulting string.
 *
 * @time_complexity O(N), where N is the length of the input string, as it iterates through the string once.
 * @space_complexity O(N), to store the resulting string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Checks if a character is a vowel.
 * @param a The character to check.
 * @return True if the character is one of 'a', 'e', 'i', 'o', 'u', 'y' (case-insensitive), False otherwise.
 */
bool isvowel(char a) {
  if (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'A' ||
      a == 'E' || a == 'I' || a == 'O' || a == 'U' || a == 'y' || a == 'Y')
    return true;
  else
    return false;
}

int main() {
  string t, ans = "";
  cin >> t;

  /**
   * @block_logic Main processing loop that transforms the input string.
   * @pre Input string `t` is read.
   * @post `ans` string is constructed based on the problem rules.
   */
  for (auto it : t) {
    if (isvowel(it)) // If the character is a vowel, do nothing.
      continue;
    else { // If the character is a consonant...
      ans += '.'; // Prepend a '.'
      ans += tolower(it); // Append the lowercase consonant.
    }
  }
  cout << ans;
}