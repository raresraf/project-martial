#include <bits/stdc++.h>

using namespace std;
string &tolower(string &s) {
  transform(s.begin(), s.end(), s.begin(), (int (*)(int))::tolower);
  return s;
}
string &remove_vowel(string &s) {
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      s.erase(i, 1);
      i--;
    }
  }
  return s;
}
bool if_consonent(char ch) {
  if (ch >= 97 && ch <= 122)
    return true;
  else
    return false;
}
string &insert_consonet(string &s) {
  int len = s.size();
  for (int i = 0; i < len; ++i) {
    cout << s.size() << endl;
    if (if_consonent(s[i]))
      s.insert(i, ".");
    ++len;
    ++i;
  }
  return s;
}
int main() {
  string s;
  cin >> s;
  s = tolower(s);
  s = remove_vowel(s);
  s = insert_consonet(s);
  cout << s << endl;
  return 0;
}
