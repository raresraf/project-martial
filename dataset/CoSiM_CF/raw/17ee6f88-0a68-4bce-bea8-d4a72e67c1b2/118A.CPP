#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  getline(cin, s);
  string out = "";
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  for (int i = 0; i < s.length(); i++) {
    char c = s[i];
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
      continue;
    out += '.';
    out += s[i];
  }
  cout << out;
}