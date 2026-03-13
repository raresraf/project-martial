/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 *
 * This program processes a string according to the following rules:
 * 1. It deletes all the vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. It inserts a character "." before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding
 *    lowercase ones.
 *
 * @see https://codeforces.com/problemset/problem/118/A
 */
#include <bits/stdc++.h>
#include <iostream>
#include <math.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  /**
   * @brief Main processing loop.
   *
   * This loop iterates through each character of the input string. It converts
   * each character to lowercase and checks if it's a consonant. If it is, it
   * prints a '.' followed by the lowercase consonant.
   */
  for (int i = 0; i < str.length(); i++) {
    char c = tolower(str[i]);
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')
      cout << "." << c;
  }
}
