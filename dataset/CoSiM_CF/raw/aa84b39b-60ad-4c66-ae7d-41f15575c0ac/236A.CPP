#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int i, c = 0;
  string s;
  cin >> s;
  sort(s.begin(), s.end());
  for (i = 0; i < s.length(); i++) {
    if (s[i] != s[i + 1]) {
      c++;
    }
  }
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}