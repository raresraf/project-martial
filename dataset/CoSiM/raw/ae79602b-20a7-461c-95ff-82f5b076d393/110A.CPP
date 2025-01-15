#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, flag = 0, n;
  char str[100];
  cin >> str;

  for (i = 0; str[i] != '\0'; i++) {
    if (str[i] == '4' || str[i] == '7')
      flag++;
  }
  if (flag == 4 || flag == 7)
    cout << "YES" << endl;

  else
    cout << "NO" << endl;
}
