/*
 *Link: https://codeforces.com/problemset/problem/236/A
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string name;
  bool abc[26];
  int diffLet = 0;
  for (int i = 0; i < 26; i++)
    abc[i] = false;
  cin >> name;
  for (int i = 0; i < name.length(); i++) {
    abc[name[i] - 'a'] = true;
  }
  for (int i = 0; i < 26; i++) {
    if (abc[i])
      diffLet++;
  }
  cout << (diffLet % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!") << endl;
}