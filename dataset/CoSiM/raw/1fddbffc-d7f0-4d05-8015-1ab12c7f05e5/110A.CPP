#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s1;
  while (cin >> s1) {
    int len1, i, value = 0, a;
    len1 = s1.length();
    for (i = 0; i < len1; i++) {
      if (s1[i] == '4' || s1[i] == '7')
        value = value + 1;
    }
    if (value == 0)
      cout << "NO" << endl;
    else {
      string res = "";
      for (i = value; i != 0; i = i / 10) {
        res.push_back(i % 10 + '0');
      }
      reverse(res.begin(), res.end());
      a = 0;
      for (i = 0; i < res.length(); i++) {
        if (res[i] == '4' || res[i] == '7')
          a++;
      }
      if (a == res.length())
        cout << "YES" << endl;
      else
        cout << "NO" << endl;
    }
  }
  return 0;
}
