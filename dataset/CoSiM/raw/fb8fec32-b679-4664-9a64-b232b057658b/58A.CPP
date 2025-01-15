#include <iostream>
#include <string.h>
//#include<ctype.h>
//#include<vector>
using namespace std;
int main() {
  string s;
  cin >> s;
  string match = "hello";
  int x = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == match[x]) {
      x++;
    }
  }
  if (x == 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}