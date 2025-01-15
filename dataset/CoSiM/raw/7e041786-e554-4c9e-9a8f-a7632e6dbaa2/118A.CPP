#include <bits/stdc++.h>
#include <string>

using namespace std;

map<char, char> mymap;

int main() {

  string s;
  string ans = "";
  char c;

  cin >> s;

  for (int i = 0; i < s.length(); ++i) {

    if (s[i] == 'a' || s[i] == 'A' || s[i] == 'e' || s[i] == 'E' ||
        s[i] == 'i' || s[i] == 'I' || s[i] == 'o' || s[i] == 'O' ||
        s[i] == 'u' || s[i] == 'U' || s[i] == 'y' || s[i] == 'Y') {

    } else {

      c = islower(s[i]) ? s[i] : tolower(s[i]);

      ans += ".";
      ans += c;
    }
  }

  cout << ans << endl;
}