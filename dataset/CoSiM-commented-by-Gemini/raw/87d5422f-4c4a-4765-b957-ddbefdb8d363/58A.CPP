/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem (Codeforces 58A).
 *
 * @details
 * The program determines if the word "hello" can be formed by deleting zero or
 * more characters from a given input string. This is equivalent to checking if
 * "hello" is a subsequence of the input string.
 *
 * The code includes several overloaded `<<` operator templates for printing
 * standard library containers, a common practice in competitive programming for
 * easier debugging, though they are not used in this solution.
 *
 * Algorithm: Greedy Two-Pointer Approach
 * 1. Define the target string "hello".
 * 2. Use a pointer/index (`cidx`) for the target string "hello", initialized to 0.
 * 3. Iterate through the input string `s` with another pointer/index `i`.
 * 4. If the character at `s[i]` matches the character at `hello[cidx]`, it means
 *    we have found the next character of the subsequence. Increment `cidx`.
 * 5. Continue until either the end of the input string `s` is reached or all
 *    characters of "hello" are found (`cidx` reaches 5).
 * 6. If `cidx` is 5 at the end, "hello" is a subsequence. Print "YES", otherwise "NO".
 *
 * Time Complexity: O(N), where N is the length of the input string, as we do a single pass.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;

// The following are template overloads for the `<<` operator, used for
// easily printing complex data structures during debugging. They are not
// used in the final solution logic for this problem.
typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;
template <typename T1, typename T2>
ostream &operator<<(ostream &os, pair<T1, T2> p) {
  os << p.first << " " << p.second;
  return os;
}
template <typename T> ostream &operator<<(ostream &os, vector<T> &v) {
  for (T i : v)
    os << i << ", ";
  return os;
}
template <typename T> ostream &operator<<(ostream &os, set<T> s) {
  for (T i : s)
    os << i << ", ";
  return os;
}
template <typename T1, typename T2>
ostream &operator<<(ostream &os, map<T1, T2> m) {
  for (pair<T1, T2> i : m)
    os << i << endl;
  return os;
}

int main() {
  string s;
  cin >> s;
  string hello = "hello";
  int cidx = 0; // Index for the target string "hello".
  /**
   * @brief Greedily searches for the "hello" subsequence in the input string 's'.
   * The loop iterates through 's', advancing the 'cidx' pointer for "hello"
   * each time a matching character is found.
   */
  for (int i = 0; i < (int)s.length() && cidx < 5; i++) {
    if (s[i] == hello[cidx]) {
      cidx++; // Match found, look for the next character of "hello".
    }
  }
  // If cidx is 5, it means all characters of "hello" were found in order.
  if (cidx >= 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
