#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  set<char> charSet;
  int length, count, i;
  cin >> str;
  length = str.size();
  for (i = 0; i < length; i++) {
    charSet.insert(str[i]);
  }
  count = charSet.size();
  if (count % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
