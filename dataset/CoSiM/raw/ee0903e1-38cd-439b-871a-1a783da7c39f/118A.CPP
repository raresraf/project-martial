#include <bits/stdc++.h>
using namespace std;
int main() {
  string ans;
  cin >> ans;
  transform(ans.begin(), ans.end(), ans.begin(), ::tolower);
  string fans;
  for (int i = 0; i < ans.length(); i++) {
    char ch = ans.at(i);
    if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
        ch == 'y') {
      continue;
    } else {
      fans = fans + '.' + ch;
    }
  }
  cout << fans << endl;
}