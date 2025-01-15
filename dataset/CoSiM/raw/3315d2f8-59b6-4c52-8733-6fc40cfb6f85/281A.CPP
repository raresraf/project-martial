#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  char x = toupper(s.at(0));
  cout << x;
  for (int i = 1; i < s.size(); i++)
    cout << s.at(i);
  return 0;
}
