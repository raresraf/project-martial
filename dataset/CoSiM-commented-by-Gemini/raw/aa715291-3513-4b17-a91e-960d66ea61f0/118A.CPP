/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem where vowels are removed, and each consonant is preceded by a '.'.
 * The implementation transforms an input string by converting all characters to lowercase,
 * filtering out vowels (a, e, i, o, u, y), and then prepending a '.' to each consonant.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. Convert the character to lowercase.
 * 4. Check if the character is a vowel.
 * 5. If it is not a vowel (i.e., a consonant), append a '.' followed by the character to the result string.
 * 6. Print the final transformed string.
 *
 * Time Complexity: O(N), where N is the length of the input string, as it requires a single pass.
 * Space Complexity: O(N), for storing the resulting string.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Checks if a character is a vowel.
 * The vowels are defined as 'a', 'e', 'i', 'o', 'u', and 'y' for this problem.
 * @param a The character to check.
 * @return True if the character is a vowel, false otherwise.
 */
bool vowel(char a) {
  // Pre-condition: 'a' is a lowercase character.
  // Check against the set of vowel characters.
  if (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'y')
    return true;
  return false;
}

int main() {
  string read;
  // Read the original string from standard input.
  cin >> read;
  string ans = "";

  /**
   * Block Logic: Iterate through the input string to build the transformed string.
   * Invariant: At each step `i`, the `ans` string contains the transformed version of the
   * first `i-1` characters of the input string.
   */
  for (int i = 0; i < read.size(); ++i) {
    // Tentatively add the '.' prefix for a consonant.
    ans += ".";
    // Convert character to lowercase for case-insensitive vowel check.
    char a = tolower(read[i]);

    /**
     * Block Logic: Determine if the character is a vowel or consonant.
     * Pre-condition: The character `a` is the lowercase version of the original character.
     */
    if (vowel(a))
      // If it's a vowel, remove the '.' that was just added.
      ans.pop_back();
    else
      // If it's a consonant, append the character itself.
      ans += a;
  }
  // Print the final transformed string to standard output.
  cout << ans << endl;
}