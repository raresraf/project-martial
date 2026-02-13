/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" problem (Codeforces 58A).
 * @author competitive-programmer
 *
 * @details This program checks if a given input string contains the word "hello"
 * as a subsequence. A subsequence means the characters 'h', 'e', 'l', 'l', 'o'
 * must appear in the input string in that specific order, but not necessarily
 * consecutively.
 *
 * Algorithm: Greedy Subsequence Matching
 * 1. Define the target string, "hello".
 * 2. Use two pointers (or indices): `i` for the input string and `j` for the
 *    target string "hello".
 * 3. Iterate through the input string with `i`.
 * 4. If the character at `str[i]` matches the character at `hello[j]`, it means
 *    we have found the j-th character of the subsequence. We then advance `j` to
 *    look for the next character of "hello".
 * 5. After iterating through the entire input string, if `j` has reached the
 *    length of "hello", it means the full subsequence was found.
 *
 * Time Complexity: O(N), where N is the length of the input string, because we
 * iterate through it only once.
 * Space Complexity: O(N) for the input string, or O(1) if input is not counted.
 */
#include <cmath>
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Reads a string and checks if "hello" is a subsequence.
 */
void solve() {
  // `str` is the input string from the user.
  string str;
  cin >> str;
  // `x` is the target subsequence we are searching for.
  string x = "hello";
  
  // `j` is an index for the target string `x`. It tracks our progress in finding the subsequence.
  int j = 0;
  
  /**
   * Block Logic: This loop iterates through the input string `str` using index `i`
   * while simultaneously tracking the found part of the subsequence with index `j`.
   * The loop continues as long as there are characters left in both the input
   * string and the target subsequence.
   */
  for (int i = 0; i < str.length() && j < x.length(); i++)
    // If the current character in the input matches the character we're looking for...
    if (str[i] == x[j])
      // ...we advance `j` to look for the next character in the "hello" subsequence.
      j++;
      
  /**
   * Decision Logic: If `j` is equal to the length of "hello", it means every
   * character of "hello" was found in the correct order.
   */
  if (j == x.length())
    cout << "YES\n";
  else
    cout << "NO\n";
}

int main() {
  solve();
  return 0;
}
