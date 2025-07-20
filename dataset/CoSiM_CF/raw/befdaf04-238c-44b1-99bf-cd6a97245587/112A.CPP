#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  int output = 0;
  for (int i = 0; i < s1.size(); i++) {
    if (int(s1[i]) == int(s2[i]))
      continue;
    if (int(s1[i]) < int(s2[i])) {
      output = -1;
      break;
    }
    if (int(s1[i]) > int(s2[i])) {
      output = 1;
      break;
    }
  }
  cout << output << endl;

  return 0;
}
