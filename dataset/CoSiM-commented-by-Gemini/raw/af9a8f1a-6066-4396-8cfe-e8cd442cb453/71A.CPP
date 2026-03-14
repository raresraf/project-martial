/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/71/A
 *
 * This program abbreviates words that are longer than 10 characters.
 */
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Processes and prints a single word according to the problem's rules.
 * @param s The word to process.
 */
void show(string s) {
  // The string is copied to a vector of characters, which is not strictly necessary
  // as the length can be checked directly on the string.
  vector<char> vec;
  for (auto c : s)
    vec.push_back(c);
  
  /**
   * @brief If the word's length is greater than 10, print its abbreviation.
   * The abbreviation consists of the first character, the number of middle
   * characters, and the last character.
   */
  if (vec.size() > 10)
    cout << s[0] << vec.size() - 2 << s[vec.size() - 1] << endl;
  /**
   * @brief Otherwise, print the word as is.
   */
  else
    cout << s << endl;
}

int main() {
  // The program first reads all words into an array.
  string arr[500];
  int num;
  cin >> num;
  for (int i = 0; i < num; i++) {
    cin >> arr[i];
    // strcpy(arr[i],s); // This commented out line is not functional C++.
  }
  
  // Then, it processes each word from the array.
  for (int i = 0; i < num; i++) {
    show(arr[i]);
  }
  return 0;
}
