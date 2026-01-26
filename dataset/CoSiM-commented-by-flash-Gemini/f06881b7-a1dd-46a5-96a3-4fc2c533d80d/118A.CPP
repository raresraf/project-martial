//
// Created by Taewoo Kim on 2/7/2017.
//

#include <iostream>
#include <regex>

using namespace std;

int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    s[i] = tolower(s[i]);
  }
  s = regex_replace(s, regex("[aeiouy]"), "");
  string res = "";
  for (char c : s) {
    res += ".";
    res += c;
  }
  cout << res;
  return 0;
}