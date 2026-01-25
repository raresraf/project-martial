#include <bits/stdc++.h>
using namespace std;
main() {
  string s;
  cin >> s;
  string t = "1111111";
  string o = "0000000";
  //	if(contains(s,t)||contains(s,o))
  if (strstr(s.c_str(), t.c_str()) || strstr(s.c_str(), o.c_str()))
    cout << "YES";
  else
    cout << "NO";
}
