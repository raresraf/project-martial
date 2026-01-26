#include <bits/stdc++.h>

using namespace std;

int main() {
  char str[101];
  int count = 0, n, i;
  cin >> str;
  n = strlen(str);
  sort(str, str + n);
  for (i = 0; str[i]; i++) {
    if (str[i] != str[i + 1]) { // kata terakhir str adalah '\0'
      count++;
    }
  }
  if (count % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
}
