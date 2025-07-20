#include <bits/stdc++.h>
using namespace std;
int main() {
  char str1[101], str2[101];
  cin >> str1 >> str2;
  int i = 0;
  while (str1[i]) {
    if (isupper(str1[i]))
      str1[i] = tolower(str1[i]);
    if (isupper(str2[i]))
      str2[i] = tolower(str2[i]);
    i++;
  }

  int result = strcmp(str1, str2);
  if (result == 0)
    cout << 0;
  else if (result < 0)
    cout << -1;
  else if (result > 0)
    cout << 1;
}
