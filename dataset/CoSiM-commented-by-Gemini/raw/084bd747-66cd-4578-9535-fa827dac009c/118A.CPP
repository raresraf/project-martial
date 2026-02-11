/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and adding a '.' before consonants.
 * @details This is a solution to the "String Task" problem from Codeforces.
 * The implementation uses a multi-stage approach with three separate loops
 * and intermediate strings to achieve the result.
 *
 * @author derekhh
 * @date May 31, 2015
 *
 * @see https://codeforces.com/problemset/problem/118/A
 *
 * Algorithm:
 * 1. An O(1) lookup table `isVowel` is initialized for quick vowel checking.
 * 2. Stage 1: Iterate through the input string. Create a new string `s1` containing
 *    only the consonants from the original string, preserving their case.
 * 3. Stage 2: Iterate through the consonant string `s1`. Create a second new
 *    string `s2` by prepending a '.' to each character.
 * 4. Stage 3: Iterate through the punctuated string `s2`. Create a third new
 *    string `s3` by converting all characters to lowercase.
 * 5. Print the final string `s3`.
 * This approach is functionally correct but inefficient due to the use of multiple
 * loops and intermediate strings where one would suffice.
 *
 * Time Complexity: O(N), where N is the length of the input string, as each of the
 * three loops is linear with respect to N.
 * Space Complexity: O(N), due to the storage required for the three intermediate
 * strings (s1, s2, s3).
 */
#include <iostream>
#include <string>
#include <cctype> // For isupper, tolower
using namespace std;

// A boolean array for O(1) vowel lookup.
bool isVowel[255];

int main() {
  // Initialize the lookup table for vowels (including 'y').
  isVowel['a'] = isVowel['e'] = isVowel['i'] = isVowel['o'] = isVowel['u'] =
      isVowel['y'] = true;
  
  string str, s1, s2, s3;
  cin >> str;

  // First pass: filter out vowels and store consonants in s1.
  for (int i = 0; i < str.size(); i++)
    if (!isVowel[tolower(str[i])])
      s1 += str[i];

  // Second pass: add a '.' before each consonant, storing in s2.
  for (int i = 0; i < s1.size(); i++) {
    s2 += '.';
    s2 += s1[i];
  }

  // Third pass: convert the entire string to lowercase, storing in s3.
  for (int i = 0; i < s2.size(); i++) {
    if (isupper(s2[i]))
      s3 += tolower(s2[i]);
    else
      s3 += s2[i];
  }

  cout << s3 << endl;
  return 0;
}
