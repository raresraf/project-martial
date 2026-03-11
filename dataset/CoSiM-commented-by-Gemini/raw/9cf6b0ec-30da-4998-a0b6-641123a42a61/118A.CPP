/**
 * @file 118A.CPP
 * @brief Processes a string by inserting dots before consonants and converting them to lowercase.
 *
 * This program implements a string manipulation task. It reads a string and transforms
 * it based on the following rules:
 * 1. Vowels ('A', 'E', 'I', 'O', 'U', 'Y', and their lowercase counterparts) are deleted.
 * 2. Each consonant is preceded by a dot '.'.
 * 3. All uppercase consonants are converted to their corresponding lowercase.
 *
 * For example, "Tour" becomes ".t.r" and "Codeforces" becomes ".c.d.f.r.c.s".
 *
 * @algorithm The program iterates through the input string character by character.
 * For each character, it checks if it is a vowel. If it's a consonant, it prints a
 * dot, converts the character to lowercase, and then prints the character.
 *
 * @time_complexity O(N), where N is the length of the string, as it processes
 * each character once.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int i; // Loop counter.
  string s; // The input string.
  cin >> s;

  /**
   * @brief Main loop to iterate through the string and apply the transformation rules.
   * @invariant The substring `s[0...i-1]` has been processed and the output printed.
   */
  for (i = 0; i < s.size(); i++) {
    // Check if the character is NOT a vowel (case-insensitive).
    if ((s[i] != 'A') && (s[i] != 'E') && (s[i] != 'I') && (s[i] != 'O') &&
        (s[i] != 'U') && (s[i] != 'Y') && (s[i] != 'a') && (s[i] != 'e') &&
        (s[i] != 'i') && (s[i] != 'o') && (s[i] != 'u') && (s[i] != 'y')) {
      
      // If it's a consonant, first print a dot.
      cout << ".";
      
      // Convert the consonant to lowercase and print it.
      // The `tolower` function correctly handles characters that are already lowercase.
      if (s[i] >= 'A' && s[i] <= 'Z') {
        s[i] = tolower(s[i]);
        cout << s[i];
      } else
        cout << s[i];
    }
  }
  return 0;
}
