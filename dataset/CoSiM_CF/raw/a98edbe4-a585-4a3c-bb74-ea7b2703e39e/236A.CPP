#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;

  int a[26];
  for (int i = 0; i < 26; i++) {
    a[i] = 0;
  }

  for (char c : str) {
    int ascii = (int)c - 97;
    a[ascii]++;
  }

  int count = 0;
  for (int i = 0; i < 26; i++) {
    if (a[i] != 0) {
      count++;
    }
  }

  count % 2 == 0 ? cout << "CHAT WITH HER!" : cout << "IGNORE HIM!";
  cout << endl;

  return 0;
}
