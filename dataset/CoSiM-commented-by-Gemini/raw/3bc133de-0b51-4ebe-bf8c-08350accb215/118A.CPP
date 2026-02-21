/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem (Codeforces 118A).
 *
 * @details The problem requires processing a string by removing vowels,
 * converting all consonants to lowercase, and inserting a '.' before each one.
 * This implementation builds the full result string before printing.
 *
 * @algorithm The program iterates through the input string `s`. A helper function,
 * `isvowel`, is used to check if a character is a vowel (case-insensitively).
 * For each character that is not a vowel:
 * 1. It is converted to its lowercase equivalent.
 * 2. A '.' character is prepended to it.
 * 3. The resulting two-character sequence is appended to a result string `ans`.
 * After iterating through the entire input string, the final `ans` string is
 * printed.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, due to a
 *                  single pass.
 * Space Complexity: O(L), to store both the input and the result strings.
 */
#include <bits/stdc++.h>

using namespace std;
// Unused template macros.
#define EPS (1e-7)
#define INF (1e9)
#define PI (acos(-1))

typedef long long ll;
#define endl '\n'

int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};

const int MOD = 1000000007;

//////////////////////////////////////////

// Unused template functions.
void Input() {}
void Solve() {}

/**
 * @brief Checks if a character is a vowel (case-insensitive, includes 'y').
 * @param c The character to check.
 * @return True if the character is a vowel, false otherwise.
 */
bool isvowel(char c) {

  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'A' ||
      c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'y' || c == 'Y') {

    return true;
  }

  return false;
}

int main() {

  // Fast I/O setup.
  cin.tie(0);
  ios::sync_with_stdio(false);
  Input(); // Calls empty template function.
  Solve(); // Calls empty template function.

  string s;
  cin >> s;

  string ans = "";

  int len = s.size();

  /**
   * @brief Block Logic: Iterates through the input string to build the output string.
   */
  for (int i = 0; i < len; i++) {

    // Process only consonants.
    if (!isvowel(s[i])) {

      // Handle lowercase consonants.
      if (s[i] >= 97 && s[i] <= 122) {

        ans += '.';
        ans += s[i];
      }
      // Handle uppercase consonants, converting to lowercase.
      if (s[i] >= 65 && s[i] <= 90) {

        ans += '.';
        ans += s[i] + 32; // ASCII-based lowercase conversion.
      }
    }
  }

  cout << ans << endl;

  return 0;
}
