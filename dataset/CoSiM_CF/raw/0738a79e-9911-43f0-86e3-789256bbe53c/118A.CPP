#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  string s;
  cin >> s;
  char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y'};
  // cout << s;
  for (int i = 0; i < s.length(); ++i) {
    if (find(vowels, vowels + 6, tolower(s[i])) == (vowels + 6)) {
      cout << '.' << (char)tolower(s[i]);
    }
  }
  cout << "\n";

  return 0;
}