#include <iostream>
#include <set>
#include <string>

using namespace std;

int main() {
  string str;
  set<char> c;

  cin >> str;

  for (int i = 0; i < str.length(); ++i) {
    c.insert(str[i]);
  }

  if (c.size() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
    return 0;
  }

  cout << "IGNORE HIM!" << endl;

  return 0;
}