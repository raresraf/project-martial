#include <bits/stdc++.h>
using namespace std;
int main() {
  char str1[102], str2[102];
  cin >> str1 >> str2;
  for (int i = 0; str1[i] != '\0'; i++) {
    str1[i] = tolower(str1[i]);
    str2[i] = tolower(str2[i]);
  }
  cout << strcmp(str1, str2) << endl;
  return 0;
}
