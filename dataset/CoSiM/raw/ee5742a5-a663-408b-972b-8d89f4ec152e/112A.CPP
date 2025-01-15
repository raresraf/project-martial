#include <bits/stdc++.h>
using namespace std;
main() {
  string s;
  string s1;
  cin >> s >> s1;
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  if (s == s1)
    cout << "0";
  else if (s < s1)
    cout << "-1";

  else
    cout << "1";
}
