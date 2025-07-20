#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  for (int i = 0; s[i] != '\0'; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      s.erase(i, 1);
      i--;
    } else {
      s.insert(i, ".");
      i++;
      s[i] = tolower(s[i]);
    }
  }
  cout << s << endl;
  return 0;
}
