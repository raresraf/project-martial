/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and formatting consonants.
 *
 * This program solves the "String Task" problem from Codeforces. It transforms an
 * input string based on the following rules:
 * 1. Deletes all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Inserts a '.' character before each consonant.
 * 3. Converts all uppercase consonants to their lowercase equivalents.
 *
 * @algorithm String Transformation / Character Filtering
 * @complexity
 * - Time: O(L * log(V)), where L is the length of the input string and V is the
 *   number of vowels in the set. The log(V) factor comes from using `std::set::find`.
 *   Since V is a small constant (6), this is effectively O(L).
 * - Space: O(L) to store the input string. The output is streamed directly, so
 *   no additional large data structure is needed for the result.
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  /**
   * @brief Functional Utility: Optimizes I/O operations for speed, a common
   * practice in competitive programming. It decouples C++ streams from C stdio
   * and unties cin from cout.
   */
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);

  // A set for efficient, case-insensitive lookup of vowels. 'y' is considered a vowel.
  set<char> vwl = {'a', 'e', 'i', 'o', 'u', 'y'};
  string s;
  cin >> s;

  /**
   * @brief Block Logic: Iterates through each character of the input string to apply the rules.
   */
  for (auto ch : s) {
    // Convert character to lowercase to handle both cases uniformly.
    ch = tolower(ch);

    // Pre-condition: Check if the character is a vowel.
    if (vwl.find(ch) != vwl.end())
      continue; // If it is a vowel, skip it (effectively deleting it).
    
    // If it's a consonant, prepend it with a '.' and print.
    cout << '.' << ch;
  }
  cout << endl;
  return 0;
}