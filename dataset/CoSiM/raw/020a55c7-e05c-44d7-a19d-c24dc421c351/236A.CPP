#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int array[26] = {0};
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    array[int(s[i]) - 'a']++;
  }
  int count = 0;
  for (int i = 0; i < sizeof(array) / sizeof(array[0]); i++) {
    if (array[i] != 0)
      count++;
  }
  if (count % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
