#include <bits/stdc++.h>
#include <cstring>
using namespace std;

int main() {
  char str[101], str2[10];
  int i, j = 0, k, l, m, n, count = 0, flag = 0;

  cin >> str;
  l = strlen(str);

  strcpy(str2, "hello");

  for (i = 0; i < l; i++) {
    if (str2[i] == str[j]) {
      j++;
      count++;

      if (count == 5) {
        flag = 1;
        break;
      }
    } else {
      i--;
      j++;
    }

    if (j == l)
      break;
  }

  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
