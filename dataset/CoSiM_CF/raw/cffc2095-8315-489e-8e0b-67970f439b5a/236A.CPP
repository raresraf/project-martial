#include <bits/stdc++.h>
using namespace std;
int main() {
  set<char> s;
  set<char>::iterator it;
  string str;
  cin >> str;
  for (int i = 0; i < str.size(); i++)
    s.insert(str[i]);

  int ck = s.size();
  if (ck % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
