#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

string replaceString(string s) {
  s.erase(std::unique(s.begin(), s.end()),
          s.end()); // remove element with characters repeated
  return s;
}
int main() {
  string n;
  cin >> n;
  sort(n.begin(),
       n.end()); // sort array so consecutive letters can get removed.
  n = replaceString(n);
  if (n.length() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}

// here is some other thing i wrote that achieves the same thing without use of
// special functions
void lol() {
  string s;
  cin >> s;
  sort(s.begin(), s.end()); // sorts by alphabet

  for (int i = 0; i < s.length(); i++) {
    if (s[i + 1] == s[i]) {
      s.erase(s.begin() + i, s.begin() + i + 1);
    }
  }
  if (s.length() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}
