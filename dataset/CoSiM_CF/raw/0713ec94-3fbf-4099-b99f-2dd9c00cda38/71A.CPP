/*
 * Link: https://codeforces.com/problemset/problem/71/A
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  int number_of_words;
  cin >> number_of_words;
  string *words = new string[number_of_words];
  for (int i = 0; i < number_of_words; i++) {
    cin >> words[i];
  }
  for (int i = 0; i < number_of_words; i++) {
    if (words[i].length() > 10) {
      words[i] = words[i][0] + to_string(words[i].length() - 2) +
                 words[i][words[i].length() - 1];
    }
    cout << words[i] << endl;
  }
}