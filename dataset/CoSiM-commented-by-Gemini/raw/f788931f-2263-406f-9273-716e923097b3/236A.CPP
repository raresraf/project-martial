#include <iostream>
#include <map>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;
  map<char, int> v;
  for (int i = 0; i < s.length(); i++) {
    v[s[i]]++;
  }
  int k = v.size();
  if (k % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}
