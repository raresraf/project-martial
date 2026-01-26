#include <cctype>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  string s1, s2;
  while (cin >> s1 >> s2) {
    for (int i = 0, j = 0; i < s1.size(), j < s2.size(); i++, j++) {
      s1[i] = tolower(s1[i]);
      s2[j] = tolower(s2[j]);
    }
    if (s1 == s2) {
      cout << "0" << endl;
    } else if (s1 > s2) {
      cout << "1" << endl;
    } else {
      cout << "-1" << endl;
    }
  }
}
