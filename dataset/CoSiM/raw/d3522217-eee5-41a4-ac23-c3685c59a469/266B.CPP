#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int t = 0, i = 0, n;
  cin >> n >> t;
  char s[n], tmp;
  while (i < n) {
    cin >> s[i];
    i++;
  }
  s[n] = '\0';
  while (t--) {
    int j = 1, k = 0;
    while (j < strlen(s)) {
      if (s[k] == 'B' && s[j] == 'G') {
        tmp = s[k];
        s[k] = s[j];
        s[j] = tmp;
        j = j + 2, k = k + 2;
      } else
        j++, k++;
    }
  }
  cout << s << endl;
  return 0;
}
