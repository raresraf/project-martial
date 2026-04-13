/**
 * @file 118A.CPP
 * @brief This program determines if two input strings can be made identical
 *        by swapping exactly two characters in one of the strings. It checks
 *        for equal length, identical character multisets, and exactly two
 *        differing positions.
 * Algorithm: Character frequency counting, String length comparison, Positional difference counting.
 * Time Complexity: O(L) where L is the length of the input strings, due to multiple
 *                  linear passes for character counting and comparison.
 * Space Complexity: O(1) for fixed-size frequency arrays (26 elements for English alphabet).
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define sz(a) (int)a.size()
#define all(c) c.begin(), c.end()
#define rep(i, n) for (int i = 0; i < (n); i++)
#define rept(i, a, b) for (int i = (a); i < (b); i++)
#define fill(a, v) memset(a, v, sizeof(a))
#define foreach(c, it)                                                         \
  for (__typeof(c.begin()) it = c.begin(); it != c.end(); ++it)

int main() {
  string a, b; // Input strings.
  // Reads the two input strings.
  cin >> a >> b;
  // chara, charb: Frequency arrays for characters in string 'a' and 'b' respectively.
  int chara[30] = {0};
  int charb[30] = {0};

  // Checks if the lengths of the two strings are different.
  if (a.length() != b.length()) {
    cout << "NO\n"; // If lengths differ, they cannot be made identical by swapping.
    return 0;
  } else {
    /**
     * Block Logic: Populates frequency array `chara` for string `a`.
     * Invariant: `chara[x]` stores the count of character 'a'+x in string `a`.
     */
    rep(i, a.length()) { // Iterates through string 'a'.
      chara[a[i] - 'a']++; // Increments count for the corresponding character.
    }
    /**
     * Block Logic: Populates frequency array `charb` for string `b`.
     * Invariant: `charb[x]` stores the count of character 'a'+x in string `b`.
     */
    rep(i, b.length()) { // Iterates through string 'b'.
      charb[b[i] - 'a']++; // Increments count for the corresponding character.
    }
    int flag = 1; // Flag to indicate if character multisets are identical (1 for yes, 0 for no).
    /**
     * Block Logic: Compares the frequency arrays to check if both strings have the same character multiset.
     * Invariant: `flag` is set to 0 if any character count differs between `chara` and `charb`.
     */
    rep(i, 26) { // Iterates for each possible lowercase letter.
      if (chara[i] != charb[i]) // If character counts differ.
        flag = 0;                // Set flag to 0.
    }

    int count = 0; // Counter for positional differences between the strings.
    if (flag == 0) { // If character multisets are not identical.
      cout << "NO\n"; // Strings cannot be made identical by swapping.
      return 0;
    } else { // If character multisets are identical.
      /**
       * Block Logic: Counts the number of positions where characters in strings `a` and `b` differ.
       * Invariant: `count` stores the total number of differing positions.
       */
      rep(i, a.length()) { // Iterates through both strings.
        if (a[i] != b[i]) // If characters at current position differ.
          count++;         // Increments positional difference counter.
      }

      // Determines if the strings can be made identical by swapping exactly two characters.
      if (count == 2) { // If exactly two positions differ.
        cout << "YES\n"; // Strings are "swappable".
      } else
        cout << "NO\n"; // Strings are not "swappable" (differ in 0, 1, or >2 positions).
    }
  }

  return 0;
}