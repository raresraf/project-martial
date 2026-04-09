/**
 * @file 118A.CPP
 * @brief This program processes an input word by converting it to a "programmers' language"
 * according to specific rules: removing vowels, inserting a period before each consonant,
 * and converting all letters to lowercase. This is a common string manipulation problem
 * in competitive programming.
 *
 * Algorithm: Iterative string processing and filtering.
 * 1. Read an input word.
 * 2. Initialize an empty string for the processed output.
 * 3. Iterate through each character of the input word:
 *    a. Convert the character to lowercase.
 *    b. Check if the lowercase character is a consonant (not 'a', 'o', 'y', 'e', 'u', 'i').
 *    c. If it's a consonant, append a '.' followed by the consonant to the output string.
 * 4. Print the resulting processed string.
 * Time Complexity: O(L) where L is the length of the input word, as each character is processed once.
 * Space Complexity: O(L) to store the new processed string.
 */

/*
  Yadi Qian
*/

#include <bits/stdc++.h>

using namespace std;

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

/**
 * @brief Main function of the program.
 * Reads a word, applies the "programmers' language" transformation rules, and prints the result.
 * @pre Input consists of a single word string.
 * @post Prints the transformed word to standard output.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Functional Utility: `word` stores the input string; `newWord` stores the processed output.
  string word;
  string newWord = "";
  // Functional Utility: Reads the input word.
  cin >> word;

  // Block Logic: Iterates through each character of the input word to apply transformation rules.
  // Invariant: `newWord` accumulates the transformed characters correctly up to the current index `i`.
  for (int i = 0; i < word.length(); i++) {
    // Functional Utility: Converts the current character to lowercase for case-insensitive processing.
    char letter = tolower(word[i]);
    // Block Logic: Checks if the lowercase character is a consonant (not a vowel).
    // Pre-condition: `letter` is a lowercase alphabetic character.
    if (letter != 'a' && letter != 'e' && letter != 'i' && letter != 'o' &&
        letter != 'u' && letter != 'y') {
      // Functional Utility: Prepends a '.' and appends the consonant to the `newWord` string.
      newWord += ".";
      newWord += letter;
    }
  }

  // Functional Utility: Prints the final transformed string.
  cout << newWord;

  return 0;
}
