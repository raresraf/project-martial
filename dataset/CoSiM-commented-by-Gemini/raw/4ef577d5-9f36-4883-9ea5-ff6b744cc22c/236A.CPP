/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces.
 *
 * @details The problem requires classifying a user as male or female based on their
 * username. If the number of distinct characters in the username is even, the
 * user is female ("CHAT WITH HER!"). If it's odd, the user is male ("IGNORE HIM!").
 *
 * @algorithm
 * This solution leverages the `std::set` container for an elegant approach.
 * 1. Read the username into a `std::string`.
 * 2. Construct a `std::set<char>` directly from the string's iterators. A `set`
 *    automatically handles uniqueness, so it will only store the distinct
 *    characters from the string.
 * 3. Check the size of the resulting set (`set.size()`). This gives the count
 *    of distinct characters.
 * 4. If the size is even, the user is female. If odd, the user is male.
 *
 * @complexity
 * Time Complexity: O(N log K), where N is the length of the input string and K
 * is the number of unique characters (at most 52 for English alphabet).
 * Each of the N characters is inserted into the set, which takes O(log K) time.
 * Since K is a small constant, this is effectively O(N).
 * Space Complexity: O(K) to store the unique characters in the set.
 */
#include <bits/stdc++.h>
using namespace std;

// A large collection of personal macros for competitive programming,
// most of which are not used in this specific solution.
#define ll long long int
#define mp make_pair
#define pb push_back
#define fi first
#define si second
#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
#define F(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cin >> a[i];                                                               \
  }
#define F1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cin >> a[i];                                                               \
  }
#define P(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
#define P1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
#define NF(a, n, m)                                                            \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
#define NF1(a, n, m)                                                           \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
#define PNF(a, n, m)                                                           \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define PNF1(a, n, m)                                                          \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define AS 200001
#define mod 1000000007

int main() {
  // `fastIO` is a macro for C++ stream I/O optimization.
  fastIO;

  string s;
  cin >> s;

  /**
   * @brief Use a set to find the number of unique characters in the string.
   * The set is constructed with iterators to the beginning and end of the
   * string `s`. The set automatically stores only the unique elements.
   */
  set<char> distinct_chars(begin(s), end(s));

  /**
   * @brief Check the parity of the count of distinct characters.
   * If the size of the set is even, the user is determined to be female.
   * Otherwise, the user is male.
   */
  if (distinct_chars.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}
