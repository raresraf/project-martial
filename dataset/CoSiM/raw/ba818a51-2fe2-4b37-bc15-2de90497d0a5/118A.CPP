#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    char c = tolower(s[i]);
    if (c != 'a' && c != 'i' && c != 'u' && c != 'e' && c != 'o' && c != 'y') {
      cout << "." << c;
    }
  }
  cout << "\n";
}
