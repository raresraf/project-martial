#include <iostream>
#include <string.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  int l = s.size();
  int k = 0, i = 0;
  for (int j = 0; j < l; j++) {
    if (s[j] == '1')
      k++;
    else
      k = 0;
    if (s[j] == '0')
      i++;
    else
      i = 0;

    if (k == 7 || i == 7) {
      cout << "YES";
      k = 7;
      i = 7;
      break;
    }
  }
  if (k < 7 || i < 7)
    cout << "NO";
}
