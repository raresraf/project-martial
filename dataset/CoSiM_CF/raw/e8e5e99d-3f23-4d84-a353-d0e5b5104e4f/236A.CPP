#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <string>

using namespace std;

int main() {
  int strSize;
  string str;
  set<char> s;
  cin >> str;
  strSize = str.size();

  for (int i = 0; i < strSize; i++) {
    s.insert(str[i]);
  }

  if (s.size() % 2 == 1) {
    cout << "IGNORE HIM!" << endl;
  } else {
    cout << "CHAT WITH HER!" << endl;
  }

  return 0;
}
