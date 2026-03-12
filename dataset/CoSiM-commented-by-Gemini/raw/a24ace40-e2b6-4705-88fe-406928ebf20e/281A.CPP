/**
 * @file 281A.CPP
 * @brief Solution to a string capitalization problem.
 *
 * This program reads a line of text and capitalizes the first character of each word.
 * A word is assumed to be a sequence of non-space characters separated by spaces.
 * This is a common task in text processing and competitive programming.
 *
 * @algorithm Linear Scan
 * @time_complexity O(N), where N is the length of the input string, because it iterates through the string once.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>

// Macro for a standard for-loop.
#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
// Macro for a reverse for-loop.
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
// Macro for iterating through a container.
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
// Macro to get an iterator range for a container.
#define _all(v) v.begin(), v.end()
// Macro to zero-out a block of memory.
#define __(v) memset(v, 0, sizeof(v))

using namespace std;
typedef long long LL;
typedef unsigned long long ULL;
/**
 * @brief Overloads the += operator for std::vector to append an element.
 * @param v The vector to which the element is appended.
 * @param x The element to append.
 * @return A reference to the modified vector.
 */
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

/**
 * @brief The core logic for capitalizing the first letter of each word in a string.
 */
void solve() {
  string s;
  std::getline(std::cin, s);
  // Capitalize the first character of the string.
  s[0] = toupper(s[0]);
  /**
   * @brief Iterates through the string to capitalize the first letter of subsequent words.
   *
   * Pre-condition: The first character of the string is already capitalized.
   * Invariant: s[i] is capitalized if it's not a space and the preceding character was a space.
   */
  _for(i, 1, s.length()) {
    if (s[i] != ' ' && s[i - 1] == ' ') {
      s[i] = toupper(s[i]);
    }
  }
  cout << s << endl;
}

int main() {
  // Optimizes I/O operations for faster execution.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
#ifdef HIEUNV
  // Redirects I/O to files for local testing, specific to the user "HIEUNV".
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  // Another I/O optimization call, often redundant but harmless.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  solve();
}