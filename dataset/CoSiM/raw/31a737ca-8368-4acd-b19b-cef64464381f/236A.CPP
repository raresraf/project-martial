#include <iostream>
#include <set>
#include <string>
using namespace std;
int main() {
  set<char> a;
  string s;
  cin >> s;
  char c;
  for (int i = 0; i < s.size(); i++) {
    c = s[i];
    a.insert(c);
  }
  if (a.size() % 2)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;
  return 0;
}
