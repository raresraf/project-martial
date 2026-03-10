/**
 * @file 118A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "String Task").
 * The program transforms a given string based on the following rules:
 * 1. All uppercase letters are converted to lowercase.
 * 2. All vowels ('a', 'e', 'i', 'o', 'u', 'y') are deleted.
 * 3. A '.' character is inserted before each consonant.
 *
 * NOTE: This file contains many boilerplate `typedef`s and headers common in
 * competitive programming, which are not used in this specific solution.
 *
 * @author Yadi Qian
 *
 * @algorithm String Transformation via Linear Scan. The program iterates through
 * the input string character by character, building a new string according to the rules.
 *
 * @time_complexity O(L), where L is the length of the input string, as it requires a single pass.
 * @space_complexity O(L), to store both the input and the potentially longer output string.
 */
/*
  Yadi Qian
*/

#include <bits/stdc++.h>

using namespace std;

// --- Boilerplate competitive programming typedefs and constants ---
const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;
// --- End of Boilerplate ---

int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string word;
  string newWord = "";
  cin >> word;

  /**
   * @block
   * @brief Iterates through the input string to build the transformed string.
   */
  for (int i = 0; i < word.length(); i++) {
    // Convert character to lowercase for case-insensitive vowel check.
    char letter = tolower(word[i]);

    // Pre-condition: Check if the character is a consonant.
    if (letter != 'a' && letter != 'e' && letter != 'i' && letter != 'o' &&
        letter != 'u' && letter != 'y') {
      // If it's a consonant, prepend '.' and append the letter to the result.
      newWord += ".";
      newWord += letter;
    }
  }

  cout << newWord;

  return 0;
}