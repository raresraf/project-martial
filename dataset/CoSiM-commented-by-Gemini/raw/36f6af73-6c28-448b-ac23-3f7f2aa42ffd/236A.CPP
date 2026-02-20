/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem by counting distinct characters in a string.
 *
 * The problem is to determine a user's gender based on their username. If the
 * number of distinct characters in the username is even, the user is female
 * ("CHAT WITH HER!"). If it's odd, the user is male ("IGNORE HIM!").
 *
 * This solution leverages `std::set` to efficiently count the distinct characters.
 *
 * Algorithm:
 * 1. Read the username into a character array.
 * 2. Construct a `std::set<char>` from the character array. The set automatically
 *    handles uniqueness, so its size will be the number of distinct characters.
 * 3. Check if the size of the set is even or odd.
 * 4. Print the corresponding message based on the result.
 *
 * Time Complexity: O(L log L), where L is the length of the string, due to the
 * complexity of inserting L elements into a balanced binary tree (the underlying
 * structure of std::set).
 * Space Complexity: O(K), where K is the number of distinct characters (at most 26
 * for the English alphabet).
 */
#include <bits/stdc++.h> // Includes set, string, iostream, etc.
using namespace std;
int main() {
  char s[100]; // Buffer for the C-style string username.
  cin >> s;

  /**
   * Block Logic: Count distinct characters using std::set.
   * A `std::set` is constructed from the character array `s`.
   * By definition, a set only stores unique elements.
   * `s1` will therefore contain each unique character from the username exactly once.
   */
  set<char> s1(s, s + strlen(s));
  
  /**
   * Block Logic: Determine gender based on the count of distinct characters.
   * The size of the set `s1` gives us the number of distinct characters.
   */
  if (s1.size() % 2 == 0) {
    // If the count is even, the user is female.
    cout << "CHAT WITH HER!";
  } else {
    // If the count is odd, the user is male.
    cout << "IGNORE HIM!";
  }
}
