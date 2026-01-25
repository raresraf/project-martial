#include <algorithm>
#include <iostream>
#include <set>

using namespace std;

int main() {

  string s;
  cin >> s;
  set<char> nchars;
  int i;
  for (i = 0; i < s.size(); i++) {
    nchars.insert(s[i]);
  }
  if (nchars.size() % 2 != 0)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
