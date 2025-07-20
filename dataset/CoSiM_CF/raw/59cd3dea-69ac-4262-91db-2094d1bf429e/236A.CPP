#include <iostream>
using namespace std;
int main() {
  string s;
  int a[26] = {0};
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    a[s[i] - 'a'] += 1;
  }
  int c = 0;
  for (int i = 0; i < 26; i++) {
    if (a[i] != 0)
      c += 1;
  }
  if (c % 2 == 0)
    cout << "CHAT WITH HER!\n";
  else
    cout << "IGNORE HIM!\n";
}