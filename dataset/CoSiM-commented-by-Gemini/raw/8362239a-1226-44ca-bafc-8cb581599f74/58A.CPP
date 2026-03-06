/**
 * @file 58A.CPP
 * @brief Determines if "hello" is a subsequence of a given string.
 *
 * This program checks if the word "hello" can be obtained by deleting zero or
 * more characters from an input string. It's a classic subsequence problem.
 *
 * @algorithm Greedy two-pointer approach.
 * @complexity
 * Time: O(N), where N is the length of the input string, due to a single pass
 *       over the input.
 * Space: O(N) to store the input string.
 */
#include <iostream>
#include <string>
#include <vector>
#include <cstring> // For memset

// --- Competitive Programming Boilerplate ---
// Custom macros for loops and other common operations to speed up coding.
#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
#define _all(v) v.begin(), v.end()
#define __(v) memset(v, 0, sizeof(v))

using namespace std;
typedef long long LL;
typedef unsigned long long ULL;
// Custom operator overload (not used in this specific solution).
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

/**
 * @brief Contains the core logic to solve the subsequence problem.
 */
void solve() {
  string s;
  cin >> s;
  string hello = "hello";
  
  // 'index' tracks our progress through the target word "hello".
  int index = 0;
  
  // Pre-condition: 'index' is 0, looking for the 'h'.
  // Block Logic: Iterate through the input string 's'.
  _for(i, 0, s.length()) {
    // If we've already found "hello", no need to check further.
    if (index == 5) break;
    
    // If the current character in 's' matches the character we're looking
    // for in "hello", advance the 'index' to look for the next character.
    if (s[i] == hello[index])
      index++;
  }
  
  // Invariant: After the loop, 'index' is the length of the longest prefix
  // of "hello" that is a subsequence of 's'.
  if (index == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}

/**
 * @brief The main function and entry point of the program.
 */
int main() {
  // Optimizations for faster I/O in competitive programming.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  
  // Conditional compilation for local testing, to redirect stdin and stdout.
  #ifdef HIEUNV
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
  #endif
  
  solve();
}
