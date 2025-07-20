#include <algorithm>
#include <iostream>
#include <string.h>
#include <string>
using namespace std;

int main() {
  string str, s2;
  cin >> str;
  char charRemove[]{'A', 'O', 'Y', 'E', 'U', 'I', 'a', 'o', 'y', 'e', 'u', 'i'};
  long n = strlen(charRemove);
  for (int j = 0; j < n; ++j) {
    str.erase(remove(str.begin(), str.end(), charRemove[j]), str.end());
  }
  long long len = str.length();
  for (int j = 0; j < len; ++j) {
    s2 += '.';
    s2 += tolower(str[j]);
  }
  cout << s2 << endl;
  ;
  return 0;
}