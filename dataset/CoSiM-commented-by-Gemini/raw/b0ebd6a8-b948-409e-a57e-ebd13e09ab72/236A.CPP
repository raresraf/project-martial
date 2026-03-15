// https://codeforces.com/problemset/problem/236/A

/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem.
 * @details This program determines a user's gender based on the number of distinct characters in their username.
 *          If the count of distinct characters is odd, the user is classified as male ("IGNORE HIM!").
 *          If the count is even, the user is classified as female ("CHAT WITH HER!").
 *
 * @algorithm
 * 1. The core logic is encapsulated in the `solve` function.
 * 2. It initializes a `std::unordered_set` with the characters from the input string.
 *    This data structure automatically stores only the unique characters.
 * 3. It then checks the size of the set.
 * 4. A bitwise AND operation (`& 1`) is used as an efficient way to test if the size is odd or even.
 * 5. Based on the result, the corresponding string ("IGNORE HIM!" or "CHAT WITH HER!") is returned.
 *
 * @complexity
 * - Time: O(L), where L is the length of the string, which is the average-case time to build the hash set.
 * - Space: O(K), where K is the number of distinct characters. In the worst case, this is proportional to the size of the character set (e.g., 26 for lowercase English letters).
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Determines the output message based on the number of distinct characters in a string.
 * @param s The input string (username).
 * @return "IGNORE HIM!" if the number of distinct characters is odd, otherwise "CHAT WITH HER!".
 */
string solve(const string &s) {
  // Construct an unordered_set from the string to get the unique characters.
  unordered_set<char> s_set(s.begin(), s.end());

  // Check if the number of unique characters is odd or even using a bitwise AND.
  if (s_set.size() & 1) {
    return "IGNORE HIM!";
  } else {
    return "CHAT WITH HER!";
  }
}

int main() {
  string s;
  // Read the username from standard input.
  cin >> s;
  // Call the solve function and print the result.
  cout << solve(s) << endl;
}