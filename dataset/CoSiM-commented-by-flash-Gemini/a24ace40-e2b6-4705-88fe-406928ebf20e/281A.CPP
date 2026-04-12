/**
 * @file 281A.CPP
 * @brief Logic for string normalization and title-case transformation.
 * 
 * Logic: Ensures that the initial character of a string and every character 
 * immediately following a whitespace is converted to its uppercase representation.
 * 
 * Algorithm: Linear sweep with single-character lookback for adjacency-based triggering.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for the string buffer.
 */

#include <bits/stdc++.h>

#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
#define _all(v) v.begin(), v.end()
#define __(v) memset(v, 0, sizeof(v))

using namespace std;
typedef long long LL;
typedef unsigned long long ULL;

// Functional Utility: Extension of the += operator for convenient vector population.
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

/**
 * @brief core transformation logic for string capitalization.
 */
void solve() {
  string s;
  std::getline(std::cin, s);
  
  /**
   * Functional Utility: Enforces uppercase on the first character of the sequence.
   * Invariant: s[0] is transformed regardless of its initial state.
   */
  s[0] = toupper(s[0]);

  /**
   * Block Logic: Scans the remaining string to resolve word boundaries.
   * Logic: Identifies transitions from whitespace to non-whitespace and applies capitalization.
   */
  _for(i, 1, s.length()) {
    if (s[i] != ' ' && s[i - 1] == ' ') {
      // Inline: Context-aware capitalization for multi-word strings.
      s[i] = toupper(s[i]);
    }
  }
  
  // Output the normalized string sequence.
  cout << s << endl;
}

int main() {
  /**
   * Functional Utility: Optimizes I/O performance for high-throughput stream processing.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(0);

#ifdef HIEUNV
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  solve();
  
  return 0;
}
