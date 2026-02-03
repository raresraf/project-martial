/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 * @see https://codeforces.com/problemset/problem/118/A
 *
 * This program processes a string according to a specific set of rules:
 * 1. It deletes all the vowels.
 * 2. It inserts a character "." before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding lowercase ones.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Create a new empty string.
 * 3. Iterate through each character of the input string.
 * 4. If the character is a consonant, append a "." to the new string, followed
 *    by the lowercase version of the consonant.
 * 5. Print the new string.
 *
 * Time Complexity: O(N), where N is the length of the string.
 * Space Complexity: O(N) to store the new string.
 */
#include <bits/stdc++.h>

using namespace std;

#define lli long long int

lli mod = 1000000007;

/**
 * @brief Checks if a character is a vowel.
 * @param c The character to check.
 * @return True if the character is a vowel, false otherwise.
 */
bool isVowel(char c) {

  char lc = tolower(c);

  if (lc == 'a' || lc == 'e' || lc == 'i' || lc == 'o' || lc == 'y' ||
      lc == 'u') {

    return true;
  }

  return false;
}

/**
 * @brief The main task of processing the string.
 */
void task() {

  string str;
  cin >> str;

  string newString = "";

  // Invariant: Iterates through the input string, building the new string
  // according to the rules.
  for (int i = 0; i < str.length(); i++) {
    // Pre-condition: If the character is not a vowel, process it.
    if (!isVowel(str[i])) {
      newString.push_back('.');
      newString.push_back(tolower(str[i]));
    }
  }

  cout << newString << '\n';
}

int main() { task(); }
