#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;

  cin >> s1;

  for (int i = 0; i < s1.length(); i++) {
    s1[i] = tolower(s1[i]);
  }

  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] != 'a' && s1[i] != 'e' && s1[i] != 'i' && s1[i] != 'o' &&
        s1[i] != 'u' && s1[i] != 'y') {
      s2.push_back(s1[i]);
    }
  }

  for (int i = 0; i < s2.length(); i++) {
    cout << "." << s2[i];
  }

  return 0;
}
