#include <bits/stdc++.h>
using namespace std;

int ans;

bool compare(char a, char b) {
  int i = (int)a;
  if (i < 97) {
    i = (i - 65 + 97);
  }

  int j = (int)b;
  if (b < 97) {
    j = (j - 65 + 97);
  }

  bool flag = true;
  if (i == j) {
    flag = false;

  } else if (i < j) {
    ans = -1;
  } else {
    ans = 1;
  }
  return flag;
}

int main() {
  string str1;
  string str2;

  cin >> str1 >> str2;

  int len = str1.length();

  for (int i = 0; i < len; i++) {
    char a = str1[i];
    char b = str2[i];

    if (compare(a, b))
      break;
    ans = 0;
  }
  cout << ans << endl;

  return 0;
}
