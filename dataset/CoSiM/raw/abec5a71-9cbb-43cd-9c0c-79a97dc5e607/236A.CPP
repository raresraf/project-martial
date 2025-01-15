#include <bits/stdc++.h>
using namespace std;

int main() {
  // your code goes here
  // You can use set
  int a[27] = {0};
  string s;
  cin >> s;
  int l = s.length();
  for (int i = 0; i < l; i++) {
    a[s[i] - 'a']++;
  }
  int cnt = 0;
  for (int i = 0; i < 26; i++) {
    if (a[i])
      cnt++;
  }
  if (cnt % 2) {
    cout << "IGNORE HIM!\n";
  } else
    cout << "CHAT WITH HER!\n";
  return 0;
}