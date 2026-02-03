/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem.
 * @see https://codeforces.com/problemset/problem/236/A
 *
 * This program determines if a user is a boy or a girl based on the number of
 * distinct characters in their username. If the number of distinct characters
 * is odd, the user is male ("IGNORE HIM!"). If it is even, the user is female
 * ("CHAT WITH HER!").
 *
 * Algorithm:
 * 1. Read the username as a string.
 * 2. Use a map to count the frequency of each character in the string. This
 *    effectively finds the number of distinct characters.
 * 3. Get the size of the map, which is the number of distinct characters.
 * 4. If the size is even, print "CHAT WITH HER!". Otherwise, print "IGNORE HIM!".
 *
 * Time Complexity: O(N), where N is the length of the username, because of the
 * single pass to populate the map.
 * Space Complexity: O(K), where K is the number of distinct characters in the
 * username (at most 26 for lowercase English letters).
 */
#include <iostream>
#include <map>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;
  map<char, int> v;
  // Invariant: Iterates through the username to count the frequency of each
  // character.
  for (int i = 0; i < s.length(); i++) {
    v[s[i]]++;
  }
  int k = v.size();
  // Block Logic: Determines the output based on the parity of the number of
  // distinct characters.
  if (k % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}