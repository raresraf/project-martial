/**
 * @file 236A.CPP
 * @brief This program analyzes an input string (e.g., a username) to determine
 *        if it contains an odd or even number of unique lowercase English letters.
 *        Based on this parity, it outputs a specific message.
 * Algorithm: Frequency Array / Counting Unique Characters.
 * Time Complexity: O(L) where L is the length of the input string, due to linear
 *                  traversals for character counting and unique character summation.
 * Space Complexity: O(1) for a fixed-size frequency array (26 elements).
 */
#include <bits/stdc++.h>
using namespace std;
#define length(a) (int)a.size()
#define all(a) a.begin(), a.end()
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define lsb(x) (x & (-x))
#define ctoi(a) (int)a - 48
#define itoc(a) a + '0'
#define MOD 1000000007
#define ll long long

int main() {
  // Functional Utility: Optimizes C++ standard input/output operations for faster execution.
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  string str;
  // Reads the input string (username).
  cin >> str;
  // Initializes an array to store the frequency of each lowercase English letter.
  int letters[26] = {0};
  /**
   * Block Logic: Iterates through the input string to populate the frequency array.
   * Invariant: For each character in the string, its corresponding count in 'letters' is incremented.
   */
  for (int i = 0; i < str.size(); i++) {
    // Inline: Increments the counter for the current lowercase letter by mapping 'a' to index 0, 'b' to 1, etc.
    letters[str[i] - 'a']++;
  }
  // Counts the number of unique characters present in the string.
  int count = 0;
  /**
   * Block Logic: Iterates through the frequency array to count how many unique letters appeared.
   * Invariant: 'count' is incremented if a letter has appeared at least once.
   */
  for (int i = 0; i < 26; i++) {
    if (letters[i])
      count++;
  }
  // Determines the output based on whether the number of unique characters is odd or even.
  if (count % 2) {
    cout << "IGNORE HIM!\n";
  } else
    cout << "CHAT WITH HER!\n";
}