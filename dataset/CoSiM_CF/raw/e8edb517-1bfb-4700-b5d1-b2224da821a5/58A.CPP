#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;
int main() {
  string str, h = "hello";
  int i, j;
  // freopen("in.txt","r",stdin);            //hell
  cin >> str;
  for (i = 0, j = 0; i < str.length(); i++) {
    if (h[j] == str[i]) {
      j++;
      if (j == 5) {
        cout << "YES";
        break;
      }
    }
  }
  if (j != 5) {
    cout << "NO";
  }

  return 0;
}
