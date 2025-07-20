#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;

  // checking substring problem

  cin >> s;

  string s1 = "0000000";
  string s2 = "1111111";

  if (strstr(s.c_str(), s1.c_str()))
    cout << "YES" << endl;
  else if (strstr(s.c_str(), s2.c_str()))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}