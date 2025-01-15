#include <iostream>
using namespace std;
int main() {
  bool letters[26] = {};
  string s;
  getline(cin, s);
  int count = 0;
  for (int i = 0; i < s.length(); i++) {
    if (!letters[s[i] - 'a']) {
      letters[s[i] - 'a'] = true;
      count++;
    }
  }
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else {
    cout << "IGNORE HIM!";
  }
}