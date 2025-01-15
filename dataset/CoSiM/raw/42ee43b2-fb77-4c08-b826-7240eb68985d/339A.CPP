#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int i;
  string s;
  string s1;
  cin >> s;
  sort(s.begin(), s.end());
  for (i = 0; i < s.length(); i++) {
    if (s[i] != '+') {
      s1.push_back(s[i]);
    }
  }
  cout << s1[0];
  for (i = 1; i < s1.length(); i++) {
    cout << "+" << s1[i];
  }

  return 0;
}