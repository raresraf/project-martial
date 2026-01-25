/******************************************
 *  Author : Debopriya Deb Roy
 *  Created On : Wed Jul 01 2020
 *  File : c++ file
 *******************************************/

#include <bits/stdc++.h>
using namespace std;
int main() {
  int i = 1, j, k, l, m, n;
  string s;
  cin >> s;
  int size = s.size();
  int count = 1, check = s[0];
  while (size--) {
    if (check == s[i]) {
      count++;
      if (count == 7)
        break;
    } else {
      count = 1;
      check = s[i];
    }
    i++;
  }
  if (count == 7)
    cout << "YES";
  else {
    cout << "NO";
  }

  return 0;
}