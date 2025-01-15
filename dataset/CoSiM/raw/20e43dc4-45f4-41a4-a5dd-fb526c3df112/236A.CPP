#include <bits/stdc++.h>
using namespace std;
int main() {
  string s1;
  set<char> s;
  cin >> s1;
  for (int i = 0; i < s1.size(); i++)
    s.insert(s1[i]);
  int ans = s.size();
  if (ans % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
