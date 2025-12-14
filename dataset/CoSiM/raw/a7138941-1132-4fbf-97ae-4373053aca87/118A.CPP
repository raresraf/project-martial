#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  string s;
  cin >> s;
  string ans = "";
  vector<char> v;
  v.push_back('a');
  v.push_back('e');
  v.push_back('i');
  v.push_back('o');
  v.push_back('u');
  v.push_back('y');
  for (int i = 0; i < s.length(); i++) {
    char temp;
    if (s[i] >= 'A' && s[i] <= 'Z')
      temp = s[i] + 32;
    else
      temp = s[i];
    if (find(v.begin(), v.end(), temp) == v.end()) {
      ans += ".";
      ans += temp;
    }
  }
  cout << ans;
  return (0);
}