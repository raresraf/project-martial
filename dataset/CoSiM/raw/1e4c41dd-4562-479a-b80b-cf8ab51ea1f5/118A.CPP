#include <bits/stdc++.h>
using namespace std;
bool is_vowel(char s) {
  if (s == 'a' || s == 'A' || s == 'O' || s == 'o' || s == 'Y' || s == 'y' ||
      s == 'E' || s == 'e' || s == 'U' || s == 'u' || s == 'i' || s == 'I')
    return true;
  return false;
}
int main() {
  string s;
  cin >> s;
  int l = s.length();
  for (int i = 0; i < s.length(); i++) {
    if (is_vowel(s[i])) {
      s.erase(i, 1);
      i--;
    } else {
      if (s[i] >= 65 && s[i] <= 90)
        s[i] += 32;
      s.insert(i, ".");
      i++;
    }
  }
  cout << s << endl;
}