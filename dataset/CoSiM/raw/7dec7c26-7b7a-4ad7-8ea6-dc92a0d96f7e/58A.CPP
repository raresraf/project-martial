#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  string test = "hello";
  int count = 0, index = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == test[index]) {
      count++;
      index++;
    }
  }
  if (count == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
