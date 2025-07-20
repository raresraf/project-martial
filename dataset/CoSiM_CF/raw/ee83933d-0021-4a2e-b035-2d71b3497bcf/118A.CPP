#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  string s;
  cin >> s;
  string ans;
  string vowels = "aeiouy";
  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
    if (vowels.find(s[i]) == string::npos) {
      ans.push_back('.');
      ans.push_back(s[i]);
    }
  }
  cout << ans;
  return 0;
}