#include <bits/stdc++.h>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;

  getline(cin, s);

  int count1 = 0;

  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      count1++;
    }
  }

  if (count1 == 4 || count1 == 7) {
    cout << "YES";
  }

  else {

    cout << "YES";
  }
}
