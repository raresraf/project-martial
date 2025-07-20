#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  int arr[26], c = 0;
  for (int i = 'a'; i <= 'z'; i++) {
    arr[i] = 0;
  }
  for (int i = 0; i < s.length(); i++) {
    arr[s[i]] = 1;
  }
  for (int i = 'a'; i <= 'z'; i++) {
    if (arr[i] != 0)
      c++;
  }
  if (c % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
