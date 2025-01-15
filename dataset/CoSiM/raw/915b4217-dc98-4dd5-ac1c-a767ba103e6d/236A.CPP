#include <algorithm>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  map<char, int> map1;
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (map1.find(s[i]) == map1.end()) {
      map1[s[i]] = i;
    }
  }

  if (map1.size() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }

  return 0;
}