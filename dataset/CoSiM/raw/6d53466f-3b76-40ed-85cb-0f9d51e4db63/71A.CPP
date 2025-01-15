#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int t;
  cin >> t;
  char s[t][101];
  int i;
  for (i = 0; i < t; i++)
    cin >> s[i];
  int len;
  for (i = 0; i < t; i++) {
    len = strlen(s[i]);
    if (len > 10) {
      cout << s[i][0] << len - 2 << s[i][len - 1] << endl;
    } else
      cout << s[i] << endl;
  }

  return 0;
}