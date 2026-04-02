/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem (Codeforces 236A).
 *
 * Problem Statement:
 * Based on a username, determine if the user is male or female. The rule is:
 * if the number of distinct characters in the username is odd, the user is male
 * ("IGNORE HIM!"). If the number of distinct characters is even, the user is
 * female ("CHAT WITH HER!").
 *
 * Algorithm:
 * The program counts the number of distinct characters in the input string 's1'.
 * It employs a nested loop structure. The outer loop iterates through each character,
 * and the inner loop checks for any subsequent occurrences of that same character.
 * A character is only counted towards the distinct count 'k' if it is the last
 * occurrence of that character in the string. This correctly determines the total
 * number of unique characters.
 *
 * Time Complexity: O(L^2), where L is the length of the string, due to the
 * nested loops. This is inefficient but functional for small constraints.
 * A more optimal solution could use a hash set or a frequency array to achieve
 * O(L) or O(L log L) complexity.
 *
 * Space Complexity: O(1), excluding the storage for the input string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s1;
  cin >> s1;
  int k = 0, l = 0; // k: counts distinct characters, l: flag for duplicates

  /**
   * Block Logic: Iterate through the string to find the number of distinct characters.
   */
  for (int i = 0; i < s1.length(); i++) {
    /**
     * Inner Block: Check if the character s1[i] appears again later in the string.
     * Pre-condition: 'l' is 0 at the start of this inner loop.
     * Invariant: 'l' is set to 1 if a duplicate of s1[i] is found.
     */
    for (int j = i + 1; j < s1.length(); j++)
      if (s1[i] == s1[j])
        l = 1;

    // If 'l' is 1, it means s1[i] is not the last occurrence of this character.
    // So, we skip counting it now and will count it only at its last position.
    if (l == 1) {
      l = 0; // Reset the flag for the next character.
      continue;
    }
    // If 'l' is 0, no duplicates were found after s1[i]. This is the last
    // occurrence, so we increment the distinct character count.
    k++;
  }

  // Based on the parity of the distinct character count, decide the output.
  if (k % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
