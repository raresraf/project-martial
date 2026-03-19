/**
 * @file raw/c68104d4-692e-4774-822b-61503b31320b/118A.CPP
 * @brief Determines if one string can be obtained from another by exactly one character swap.
 *
 * This program checks if two input strings, `a` and `b`, are "swappable".
 * Two strings are considered swappable if one can be transformed into the other
 * by swapping exactly two of its characters.
 *
 * @algorithm
 * 1.  **Length Check:** If the strings have different lengths, they cannot be
 *     made equal with a swap. The program exits with "NO".
 * 2.  **Anagram Check:** The strings must be anagrams of each other. This is
 *     verified by counting the frequency of each character in both strings. If
 *     the character counts do not match, the program exits with "NO".
 * 3.  **Difference Count:** If the strings are anagrams, the program counts
 *     the number of positions where the characters are different.
 * 4.  **Final Condition:** A single swap can fix at most two differing positions.
 *     Therefore, if the strings are anagrams and differ at exactly two
 *     positions, they are one swap apart. The program outputs "YES". Otherwise
 *     (0 or >2 differences), it outputs "NO".
 *
 * @time_complexity O(L), where L is the length of the strings, as the process
 *                  involves a few linear passes over the strings.
 * @space_complexity O(1), as the frequency arrays have a constant size (26 for the alphabet).
 */
#include <bits/stdc++.h>
using namespace std;

// --- Competitive Programming Macros ---
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
#define foreach(c, it)                                                         
  for (__typeof(c.begin()) it = c.begin(); it != c.end(); ++it)

int main() {
  string a, b;
  cin >> a >> b;
  int chara[30] = {0};
  int charb[30] = {0};

  // Block: Initial sanity check for length equality.
  if (a.length() != b.length()) {
    cout << "NO
";
    return 0;
  } else {
    // Block: Anagram check using frequency counting.
    rep(i, a.length()) { chara[a[i] - 'a']++; }
    rep(i, b.length()) { charb[b[i] - 'a']++; }
    int flag = 1; // Assume they are anagrams.
    rep(i, 26) {
      if (chara[i] != charb[i])
        flag = 0; // Found a mismatch, they are not anagrams.
    }

    if (flag == 0) {
      cout << "NO
";
      return 0;
    } else {
      // Block: Count differing positions in the anagrams.
      int count = 0;
      rep(i, a.length()) {
        if (a[i] != b[i])
          count++;
      }

      // Block: Final check. If they are anagrams, they must differ by
      // exactly 2 characters to be solvable with one swap. An identical
      // string (count = 0) is not a valid solution.
      if (count == 2) {
        cout << "YES
";
      } else
        cout << "NO
";
    }
  }

  return 0;
}
