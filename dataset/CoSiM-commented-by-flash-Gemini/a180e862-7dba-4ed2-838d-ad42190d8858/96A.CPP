/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces (Problem 96A).
 * @details The program determines if a situation is "dangerous" in a football match.
 * A situation is considered dangerous if there are at least seven consecutive
 * players of the same team.
 *
 * @algorithm
 * 1. Read the input string representing the players on the field ('0's and '1's).
 * 2. Define two target strings: "1111111" and "0000000".
 * 3. Use a substring search function to check if either of these target strings
 *    exists within the input string.
 * 4. If either substring is found, the situation is dangerous, so print "YES".
 * 5. If neither is found after checking the entire input string, print "NO".
 *
 * @time_complexity O(N*M), where N is the length of the input string and M is the
 * length of the substring to find (7). Since M is constant, it simplifies to O(N).
 * @space_complexity O(1), besides the storage for the input string.
 */
#include <bits/stdc++.h>
#include <string>
using namespace std;

/**
 * @brief Checks if a string `s1` is a substring of another string `s2`.
 * @param s1 The potential substring.
 * @param s2 The string to be searched within.
 * @return The starting index of `s1` in `s2` if found, otherwise -1.
 */
int isSubstring(string s1, string s2) {
  int M = s1.length();
  int N = s2.length();
  // A loop to slide s1 one by one.
  for (int i = 0; i <= N - M; i++) {
    int j;
    // For current index i, check for pattern match.
    for (j = 0; j < M; j++)
      if (s2[i + j] != s1[j])
        break;

    if (j == M)
      return i;
  }
  return -1;
}

int main() {
  string s; // Input string of player positions.
  string s1 = "1111111"; // Represents 7 consecutive players of team 1.
  string s0 = "0000000"; // Represents 7 consecutive players of team 0.
  getline(cin, s);

  /**
   * @block_logic Check for the existence of the dangerous patterns.
   */
  int res1 = isSubstring(s1, s);
  int res0 = isSubstring(s0, s);
  
  // If neither '1111111' nor '0000000' is found, the situation is not dangerous.
  if (res1 == -1 && res0 == -1)
    cout << "NO";
  else // Otherwise, it is dangerous.
    cout << "YES";
  return 0;
}
