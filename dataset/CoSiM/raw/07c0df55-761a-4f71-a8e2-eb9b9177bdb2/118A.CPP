#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  for (auto &c : s) {
    c = tolower(c);
  }

  s.erase(std ::remove(s.begin(), s.end(), 'A'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'a'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'E'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'e'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'I'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'i'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'O'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'o'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'U'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'u'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'Y'), s.end());
  s.erase(std ::remove(s.begin(), s.end(), 'y'), s.end());

  for (int i = 0; i <= s.size(); i++) {

    if (i == s.size()) {
      break;
    } else {
      cout << "." << s[i];
    }
  }

  return 0;
}
