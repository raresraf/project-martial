#include <bits/stdc++.h>
using namespace std;
main() {
  string s;
  cin >> s;
  int count = 0;
  map<char, int> p;
  int flag = 0;
  //	if(islower(s[0]))
  //	 {
  //	 	flag=1;
  //
  //	 }
  //	 else
  //	 {
  for (int i = 0; i < s.size(); i++) {

    p[s[i]]++;
  }

  for (int i = 0; i < s.size(); i++) {
    if (p[s[i]]) {
      count++;
    }
    p[s[i]] = 0;
  }

  //    }

  if (count % 2 != 0)
    flag = 1;

  if (flag == 1) {
    cout << "IGNORE HIM!";
  }

  else
    cout << "CHAT WITH HER!";
}
