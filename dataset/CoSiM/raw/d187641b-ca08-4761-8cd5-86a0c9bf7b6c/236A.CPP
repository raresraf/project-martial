#include <iostream>
using namespace std;
int main() {
  int num[26] = {0}, distinct = 0;
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); ++i) {
    if (!num[s[i] - 'a']) {
      distinct += 1;
      num[s[i] - 'a'] += 1;
    }
  }
  cout << ((distinct % 2) ? "IGNORE HIM!" : "CHAT WITH HER!");
  return 0;
}
