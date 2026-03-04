/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - "Way Too Long Words".
 *
 * This program shortens words that are longer than 10 characters. The abbreviation
 * is formed by the word's first character, the number of characters between the
 * first and last, and the last character. Shorter words are left unchanged.
 *
 * @algorithm String length check and conditional abbreviation.
 * @time_complexity O(N*L), where N is the number of words and L is the length
 *                  of the longest word. The `newword` function has a loop that
 *                  is linear in the length of the string, making the overall
 *                  complexity dependent on both N and L.
 * @space_complexity O(L) to store the longest word and its character vector.
 */
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define pb push_back
#define vc vector<char>

/**
 * @brief Abbreviates a word if its length is greater than 10.
 * @param s The input string.
 * @return The abbreviated string or the original string.
 *
 * This function handles the logic for word abbreviation. The implementation is
 * somewhat verbose, as it converts the string to a character vector and uses
 * a loop to count the intermediate characters, a value that could also be
 * calculated directly as `s.length() - 2`.
 */
string newword(string s) {
  vc a(s.begin(), s.end());

  int p = a.size();
  string ans = "";
  if (p > 10) {
    // long word
    char start = a[0];
    char end = a[a.size() - 1];
    int count = 0;
    // This loop is equivalent to `count = a.size() - 2;`
    for (size_t i = 1; i < a.size() - 1; i++) {
      count += 1;
    }
    ans += start + to_string(count) + end;
  } else {
    // return the string as it is
    ans += s;
  }
  return ans;
}

int main() {
  int n;
  cin >> n;
  /**
   * @brief Main loop to process each word.
   * Reads 'n' words and prints the result of the `newword` function for each.
   */
  while (n--) {
    string s;
    cin >> s;
    cout << newword(s) << "\n";
  }
  return 0;
}