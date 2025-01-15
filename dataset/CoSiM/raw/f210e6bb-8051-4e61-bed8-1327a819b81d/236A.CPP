#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  map<char, int> in;
  for (int i = 0; i < s.length(); i++) {
    in[s[i]]++;
  }
  int x = in.size();
  if (x % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
